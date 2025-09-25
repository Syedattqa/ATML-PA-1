import argparse, os, json, random, math
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode, functional as TF
from torchvision.utils import make_grid, save_image
import timm
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# -------------------
# CLI
# -------------------
p = argparse.ArgumentParser("STL10 invariance/robustness eval (no retrain)")
p.add_argument("--data_root", default="./data")
p.add_argument("--weights_dir", default=r"M:\ATML\PA1\out_stl10_focus")
p.add_argument("--resnet_ckpt", default="resnet50_best.pt")
p.add_argument("--vit_ckpt",    default="vit_s16_best.pt")
p.add_argument("--batch_size", type=int, default=128)
p.add_argument("--workers", type=int, default=0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--cache_dir", default="cache_eval")
# transforms
p.add_argument("--shift_pixels", type=int, nargs="+", default=[2,4,8], help="pixels to shift")
p.add_argument("--shuffle_grids", type=int, nargs="+", default=[2,4,16], help="NxN grids")
p.add_argument("--occ_sizes", type=float, nargs="+", default=[0.25, 0.4], help="occlusion size as frac of side")
p.add_argument("--occ_mode", choices=["mask","blur"], default="mask")
p.add_argument("--occ_place", choices=["center","random"], default="center")
# visualization
p.add_argument("--viz_examples", type=int, default=8)
args = p.parse_args()

SEED = args.seed
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.cuda.is_available()
IM_SIZE = 224
NUM_CLASSES = 10
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CLASS_NAMES = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']

OUT = Path("out_eval"); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(args.cache_dir); CACHE.mkdir(parents=True, exist_ok=True)

# -------------------
# Data
# -------------------
def tf_test():
    return transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def load_test():
    return datasets.STL10(root=args.data_root, split="test", download=False, transform=tf_test())

# -------------------
# Models & ckpt load
# -------------------
def make_resnet50(): return timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
def make_vit_s16():  return timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=NUM_CLASSES)

def load_model(tag:str, ckpt_path:Path):
    model = make_resnet50() if tag=="resnet50" else make_vit_s16()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(DEVICE).eval()
    return model

# -------------------
# Utilities
# -------------------
def denorm(x: torch.Tensor) -> torch.Tensor:
    """
    Works for 3D (C,H,W) or 4D (N,C,H,W). Returns same rank as input.
    """
    if x.dim() == 3:
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3,1,1)
        std  = torch.tensor(IMAGENET_STD,  device=x.device).view(3,1,1)
        return (x*std + mean).clamp(0,1)
    elif x.dim() == 4:
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1,3,1,1)
        std  = torch.tensor(IMAGENET_STD,  device=x.device).view(1,3,1,1)
        return (x*std + mean).clamp(0,1)
    else:
        raise RuntimeError(f"denorm expects 3D or 4D, got shape {tuple(x.shape)}")


def softmax_conf(x:torch.Tensor)->torch.Tensor:
    return torch.softmax(x, dim=1).amax(dim=1)

def to_chw01(vis: torch.Tensor) -> torch.Tensor:
    """
    Ensure the tensor is (3,H,W) in [0,1].
    Accepts CHW or HWC, with or without a leading batch of size 1.
    """
    t = vis
    # drop accidental batch
    if t.dim() == 4 and t.size(0) == 1:
        t = t.squeeze(0)

    # If HWC, permute to CHW
    if t.dim() == 3 and t.shape[-1] == 3 and t.shape[0] != 3:
        t = t.permute(2,0,1).contiguous()  # HWC -> CHW

    # Now expect CHW
    assert t.dim() == 3 and t.size(0) == 3, f"Unexpected vis shape {tuple(t.shape)}"

    # Clamp to [0,1] if not already
    t = t.clamp(0,1)
    return t


# -------------------
# Transform datasets with caching
# -------------------
class TransformedSTL(Dataset):
    """
    Wraps STL10 test and applies a transformation on-the-fly OR loads a cached tensor file.
    Cache format: dict{ 'images': float32 Nx3xHxW normalized, 'labels': int64 }
    """
    def __init__(self, base:datasets.STL10, name:str, build_fn):
        self.base = base
        self.name = name
        self.cache_path = CACHE / f"{name}.pt"
        if self.cache_path.exists():
            blob = torch.load(self.cache_path, map_location="cpu")
            self.imgs = blob["images"]; self.labels = blob["labels"]
        else:
            xs=[]; ys=[]
            for i in tqdm(range(len(base)), desc=f"build-{name}"):
                x,y = base[i]
                x = build_fn(x)  # x still normalized after transform
                if x.dim() == 4 and x.size(0) == 1:  # strip accidental batch dim
                    x = x.squeeze(0)
                assert x.dim() == 3, f"built sample has shape {tuple(x.shape)}"




                xs.append(x.cpu()); ys.append(int(y))
            self.imgs = torch.stack(xs).float()
            self.labels = torch.tensor(ys, dtype=torch.long)
            torch.save({"images":self.imgs, "labels":self.labels}, self.cache_path)
        self.N = self.labels.numel()
    def __len__(self): return self.N
    def __getitem__(self, i):
        return self.imgs[i], self.labels[i]

# --- Specific transforms (operate on normalized tensors) ---
def t_shift(dx:int, dy:int):
    # reflect padding to avoid black borders; operate in [0,1], then renorm
    def fn(xn):
        x= denorm(xn)  # -> [0,1]
        pad = (abs(dx), abs(dx), abs(dy), abs(dy))
        x = TF.pad(x, pad, padding_mode="reflect")
        x = TF.affine(x, angle=0, translate=(dx, dy), scale=1.0, shear=[0.0,0.0], interpolation=InterpolationMode.BILINEAR)
        # crop back to original size
        x = TF.center_crop(x, [IM_SIZE, IM_SIZE])
        # re-normalize
        mean = torch.tensor(IMAGENET_MEAN).view(3,1,1); std=torch.tensor(IMAGENET_STD).view(3,1,1)

        x = (x - mean) / std

        if x.dim() == 4 and x.size(0) == 1:
            x = x.squeeze(0)
        assert x.dim() == 3, f"t_shift produced {tuple(x.shape)}"
        return x
    return fn

def t_patch_shuffle(grid:int):
    # split into grid x grid tiles and permute
    def fn(xn):
        x = denorm(xn)
        C,H,W = x.shape
        th, tw = H//grid, W//grid
        tiles=[]
        for r in range(grid):
            for c in range(grid):
                tiles.append(x[:, r*th:(r+1)*th, c*tw:(c+1)*tw])
        order = torch.randperm(len(tiles))
        tiles = [tiles[i] for i in order]
        rows=[]
        for r in range(grid):
            rows.append(torch.cat(tiles[r*grid:(r+1)*grid], dim=2))
        x = torch.cat(rows, dim=1)
        mean = torch.tensor(IMAGENET_MEAN).view(3,1,1); std=torch.tensor(IMAGENET_STD).view(3,1,1)
        x=(x - mean)/std

        if x.dim() == 4 and x.size(0) == 1:
            x = x.squeeze(0)
        assert x.dim() == 3, f"t_shift produced {tuple(x.shape)}"
        return x
    return fn

def gaussian_blur(x, k=11, sigma=5.0):
    # depthwise 2D gaussian
    import math
    grid = torch.arange(k) - (k-1)/2
    g = torch.exp(-(grid**2)/(2*sigma**2)); g = g / g.sum()
    kernel = torch.outer(g,g)
    kernel = kernel.view(1,1,k,k).to(x.device)
    kernel = kernel.repeat(x.shape[0],1,1,1)
    x = F.conv2d(x.unsqueeze(0), kernel, padding=k//2, groups=x.shape[0]).squeeze(0)
    return x

def t_occlude(frac:float, mode:str="mask", place:str="center"):
    side = int(IM_SIZE * frac)
    def fn(xn):
        x = denorm(xn)
        C,H,W = x.shape
        if place=="center":
            top = (H - side)//2; left = (W - side)//2
        else:
            top = random.randint(0, H-side)
            left = random.randint(0, W-side)
        if mode=="mask":
            # gray mask ~ dataset mean
            color = torch.tensor(IMAGENET_MEAN).view(3,1,1)
            x[:, top:top+side, left:left+side] = color
        else:
            patch = x[:, top:top+side, left:left+side]
            x[:, top:top+side, left:left+side] = gaussian_blur(patch, k=15, sigma=6.0)
        mean = torch.tensor(IMAGENET_MEAN).view(3,1,1); std=torch.tensor(IMAGENET_STD).view(3,1,1)
        x=(x - mean)/std
        if x.dim() == 4 and x.size(0) == 1:
            x = x.squeeze(0)
        assert x.dim() == 3, f"t_shift produced {tuple(x.shape)}"
        return x

    return fn

# -------------------
# Eval helpers
# -------------------
@dataclass
class EvalOut:
    acc: float
    consist: float
    avg_conf: float
    avg_conf_drop: float
    n: int

@torch.no_grad()
def eval_set(model:nn.Module, loader:DataLoader, clean_logits:torch.Tensor)->EvalOut:
    # clean_logits: NxK logits for consistency/ drop (aligned with dataset order)
    model.eval()
    total=0; correct=0
    conf_sum=0.0; drop_sum=0.0; consist=0
    idx = 0
    for x,y in tqdm(loader, leave=False):
        x=x.to(DEVICE); y=y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(1)
        conf = torch.softmax(logits, dim=1).amax(dim=1)
        # slice corresponding clean logits
        bs = x.size(0)
        clean_slice = clean_logits[idx:idx+bs].to(DEVICE)
        clean_pred  = clean_slice.argmax(1)
        clean_conf  = torch.softmax(clean_slice, dim=1).amax(dim=1)
        # stats
        correct += (preds==y).sum().item()
        total += bs
        conf_sum += conf.sum().item()
        drop_sum += (clean_conf - conf).sum().item()
        consist += (preds == clean_pred).sum().item()
        idx += bs
    return EvalOut(acc=correct/total, consist=consist/total,
                   avg_conf=conf_sum/total, avg_conf_drop=drop_sum/total, n=total)

@torch.no_grad()
def logits_over_set(model, loader)->torch.Tensor:
    model.eval(); outs=[]
    for x,_ in tqdm(loader, leave=False):
        x=x.to(DEVICE)
        outs.append(model(x).cpu())
    return torch.cat(outs,0)

# -------------------
# Figures & tables
# -------------------
# -------------------
# Figures & tables
# -------------------
def save_examples_grid(clean_ds: TransformedSTL, trans_ds: TransformedSTL,
                       res_model, vit_model, save_path: Path, k: int = 8):

    assert k > 0, "k must be >= 1"
    idxs = np.linspace(0, len(clean_ds) - 1, num=k, dtype=int)

    tiles: List[torch.Tensor] = []
    captions: List[str] = []

    def _to_vis(t: torch.Tensor) -> torch.Tensor:
        # -> CHW in [0,1]
        t = denorm(t.cpu())
        return to_chw01(t)

    res_model.eval(); vit_model.eval()
    with torch.no_grad():
        for i in idxs:
            xc, y = clean_ds[i]
            xt, _ = trans_ds[i]

            xc_vis = _to_vis(xc)
            xt_vis = _to_vis(xt)

            xr = xc.unsqueeze(0).to(DEVICE)
            xtt = xt.unsqueeze(0).to(DEVICE)

            # predictions
            lr_c = res_model(xr); lr_t = res_model(xtt)
            lv_c = vit_model(xr); lv_t = vit_model(xtt)

            prc = CLASS_NAMES[lr_c.argmax(1).item()]
            pvc = CLASS_NAMES[lv_c.argmax(1).item()]
            prt = CLASS_NAMES[lr_t.argmax(1).item()]
            pvt = CLASS_NAMES[lv_t.argmax(1).item()]

            cr  = float(softmax_conf(lr_c)[0]); cv = float(softmax_conf(lv_c)[0])
            tr  = float(softmax_conf(lr_t)[0]); tv = float(softmax_conf(lv_t)[0])

            # order: clean tile (bottom caption), transformed tile (top caption)
            tiles += [xc_vis, xt_vis]
            captions.append(f"C-R:{prc} ({cr:.2f})  V:{pvc} ({cv:.2f})")  # bottom strip
            captions.append(f"T-R:{prt} ({tr:.2f})  V:{pvt} ({tv:.2f})")  # top strip

    # ---- assemble grid image ----
    PAD = 8  # must match torchvision.utils.make_grid padding
    grid = make_grid(torch.stack(tiles, dim=0), nrow=2, padding=PAD)  # shape: (C, H, W)

    # fig size scales with number of rows (k)
    fig_h = 3.2 * (k / 2) if k >= 2 else 2.8
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
    ax.axis('off')
    ax.set_title("Clean vs Transformed (predictions | confidences)")

    # ---- geometry per-tile ----
    rows, cols = k, 2                    # since we stacked tiles in pairs
    tile_h = tiles[0].shape[1]           # e.g., 224
    tile_w = tiles[0].shape[2]           # e.g., 224
    strip_h = 26                         # caption strip height
    fs = 9                               # font size

    # simple wrapper to avoid overly long lines
    def wrap_text(s: str, max_chars: int = 30) -> str:
        if len(s) <= max_chars:
            return s
        parts = []
        while len(s) > max_chars:
            cut = s.rfind(' ', 0, max_chars)
            cut = cut if cut != -1 else max_chars
            parts.append(s[:cut].strip())
            s = s[cut:].lstrip()
        if s:
            parts.append(s)
        return "\n".join(parts)

    # ---- draw captions: TOP for transformed ("T-..."), BOTTOM for clean ("C-...") ----
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            cap = captions[idx]
            x0 = PAD + c * (tile_w + PAD)
            y0 = PAD + r * (tile_h + PAD)

            if cap.startswith("T-"):
                # TOP strip (transformed)
                ax.add_patch(plt.Rectangle((x0, y0), tile_w, strip_h,
                                           linewidth=0, facecolor='black', alpha=0.85))
                ax.text(x0 + 5, y0 + 4, wrap_text(cap),
                        fontsize=fs, color='white', va='bottom', ha='left')
            else:
                # BOTTOM strip (clean)
                ax.add_patch(plt.Rectangle((x0, y0 + tile_h - strip_h),
                                           tile_w, strip_h,
                                           linewidth=0, facecolor='black', alpha=0.85))
                ax.text(x0 + 5, y0 + tile_h - 6, wrap_text(cap),
                        fontsize=fs, color='white', va='top', ha='left')

    # ---- keep edges safe; no tight bbox that crops labels ----
    ax.margins(x=0.0, y=0.02)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.06)

    plt.savefig(save_path, dpi=200)  # do NOT use bbox_inches="tight"
    plt.close(fig)

def save_table(rows:List[Dict], title:str, png_path:Path, csv_path:Path):
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    fig,ax = plt.subplots(figsize=(10, 0.4*len(rows)+1.6)); ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.2)
    plt.title(title, pad=10)
    plt.savefig(png_path, dpi=200, bbox_inches='tight'); plt.close()

# -------------------
# Main
# -------------------
def main():
    print(f"Device: {DEVICE} (AMP={AMP})")
    test_ds = load_test()
    base = TransformedSTL(test_ds, "clean", lambda x:x)
    base_loader = DataLoader(base, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # load models
    res_model = load_model("resnet50", Path(args.weights_dir)/args.resnet_ckpt)
    vit_model = load_model("vit_s16",  Path(args.weights_dir)/args.vit_ckpt)

    # clean logits (for consistency baselines)
    print("Computing clean logits...")
    res_clean_logits = logits_over_set(res_model, base_loader)
    vit_clean_logits = logits_over_set(vit_model, base_loader)

    results_rows = []

    # ---- Translation shifts ----
    for s in args.shift_pixels:
        for (dx,dy) in [(s,0),(-s,0),(0,s),(0,-s),(s,s),(-s,-s),(s,-s),(-s,s)]:
            name = f"shift_dx{dx}_dy{dy}"
            ds = TransformedSTL(test_ds, name, t_shift(dx,dy))
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            res = eval_set(res_model, loader, res_clean_logits)
            vit = eval_set(vit_model, loader, vit_clean_logits)
            results_rows.append({"Transform":name,"Model":"ResNet-50","Acc":f"{res.acc:.3f}","Consist":f"{res.consist:.3f}",
                                 "AvgConf":f"{res.avg_conf:.3f}","AvgConfDrop":f"{res.avg_conf_drop:.3f}","N":res.n})
            results_rows.append({"Transform":name,"Model":"ViT-S/16","Acc":f"{vit.acc:.3f}","Consist":f"{vit.consist:.3f}",
                                 "AvgConf":f"{vit.avg_conf:.3f}","AvgConfDrop":f"{vit.avg_conf_drop:.3f}","N":vit.n})
        # one example figure per magnitude
        ds = TransformedSTL(test_ds, f"shift_dx{s}_dy{0}", t_shift(s,0))
        save_examples_grid(base, ds, res_model, vit_model, OUT/f"viz_shift_{s}px.png", k=args.viz_examples)

    # ---- Patch shuffling ----
    for g in args.shuffle_grids:
        name = f"shuffle_{g}x{g}"
        ds = TransformedSTL(test_ds, name, t_patch_shuffle(g))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        res = eval_set(res_model, loader, res_clean_logits)
        vit = eval_set(vit_model, loader, vit_clean_logits)
        results_rows.append({"Transform":name,"Model":"ResNet-50","Acc":f"{res.acc:.3f}","Consist":f"{res.consist:.3f}",
                             "AvgConf":f"{res.avg_conf:.3f}","AvgConfDrop":f"{res.avg_conf_drop:.3f}","N":res.n})
        results_rows.append({"Transform":name,"Model":"ViT-S/16","Acc":f"{vit.acc:.3f}","Consist":f"{vit.consist:.3f}",
                             "AvgConf":f"{vit.avg_conf:.3f}","AvgConfDrop":f"{vit.avg_conf_drop:.3f}","N":vit.n})
        save_examples_grid(base, ds, res_model, vit_model, OUT/f"viz_shuffle_{g}x{g}.png", k=args.viz_examples)

    # ---- Patch occlusion ----
    for frac in args.occ_sizes:
        name = f"occlude_{args.occ_mode}_{args.occ_place}_{int(frac*100)}p"
        ds = TransformedSTL(test_ds, name, t_occlude(frac, args.occ_mode, args.occ_place))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        res = eval_set(res_model, loader, res_clean_logits)
        vit = eval_set(vit_model, loader, vit_clean_logits)
        results_rows.append({"Transform":name,"Model":"ResNet-50","Acc":f"{res.acc:.3f}","Consist":f"{res.consist:.3f}",
                             "AvgConf":f"{res.avg_conf:.3f}","AvgConfDrop":f"{res.avg_conf_drop:.3f}","N":res.n})
        results_rows.append({"Transform":name,"Model":"ViT-S/16","Acc":f"{vit.acc:.3f}","Consist":f"{vit.consist:.3f}",
                             "AvgConf":f"{vit.avg_conf:.3f}","AvgConfDrop":f"{vit.avg_conf_drop:.3f}","N":vit.n})
        save_examples_grid(base, ds, res_model, vit_model, OUT/f"viz_occlude_{int(frac*100)}p_{args.occ_mode}_{args.occ_place}.png", k=args.viz_examples)

    # ---- Save combined table
    save_table(results_rows,
               "STL-10: Translation / Patch Shuffling / Occlusion (Acc, Consistency, Confidence)",
               OUT/"invariance_results.png",
               OUT/"invariance_results.csv")

    print("All artifacts in:", OUT.resolve())
    print("Cached transformed sets in:", CACHE.resolve())

if __name__ == "__main__":
    main()
