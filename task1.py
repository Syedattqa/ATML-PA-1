# file: stl10_stylized_gray_bias.py
import os, platform, argparse, json, random, time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid, save_image
import timm
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.metrics import structural_similarity as ssim

# ---- AdaIN (official) ----
from gitadain import decoder as adain_decoder, vgg as adain_vgg
from gitadain import adaptive_instance_normalization as adain

# ---------- Config ----------
p = argparse.ArgumentParser()
p.add_argument("--data_root", type=str, default="./data")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--lr_resnet", type=float, default=3e-4)
p.add_argument("--lr_vit", type=float, default=5e-4)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--workers", type=int, default=None)
p.add_argument("--alpha", type=float, default=1.0, help="AdaIN blend (1 = strong style)")
p.add_argument("--quick", action="store_true", help="1-epoch smoke test")
args = p.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.cuda.is_available()
if args.workers is None:
    args.workers = 0 if platform.system().lower().startswith("win") else 4

SEED = args.seed
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

OUT = Path("out_stl10_focus"); OUT.mkdir(parents=True, exist_ok=True)
IM_SIZE = 224
NUM_CLASSES = 10
CLASS_NAMES = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Paths to official AdaIN weights
ADAIN_VGG = r"M:\ATML\PA1\weights\vgg_normalised.pth"
ADAIN_DEC = r"M:\ATML\PA1\weights\decoder.pth"

print(f"Using device: {DEVICE} (AMP={AMP}), workers={args.workers}")

# ---------- Transforms ----------
def tf_train():
    return transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def tf_test():
    return transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def tf_gray3():
    return transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

# ---------- Data ----------
def prepare_stl10(root="./data"):
    train = datasets.STL10(root=root, split="train", download=False, transform=tf_train())
    test  = datasets.STL10(root=root, split="test",  download=False, transform=tf_test())
    n_val = int(0.2*len(train)); n_train = len(train)-n_val
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(train, [n_train, n_val], generator=g)
    return train_ds, val_ds, test

# ---------- Models ----------
def make_resnet50(): return timm.create_model("resnet50", pretrained=True, num_classes=NUM_CLASSES)
def make_vit_s16():  return timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=NUM_CLASSES)

@dataclass
class TrainCfg:
    epochs:int; batch:int; lr:float; workers:int

def accuracy(logits, y): return (logits.argmax(1)==y).float().mean().item()

def train_eval(model, cfg:TrainCfg, train_ds, val_ds, test_ds, tag:str):
    tr_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True,  num_workers=cfg.workers, pin_memory=True)
    va_loader = DataLoader(val_ds,   batch_size=cfg.batch, shuffle=False, num_workers=cfg.workers, pin_memory=True)
    te_loader = DataLoader(test_ds,  batch_size=cfg.batch, shuffle=False, num_workers=cfg.workers, pin_memory=True)

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    best, best_path = -1.0, OUT/f"{tag}_best.pt"
    print(f"=== TRAIN {tag} ===")
    for ep in range(cfg.epochs):
        model.train(); accs=[]
        for x,y in tqdm(tr_loader, desc=f"train-{tag}"):
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            if AMP:
                with torch.cuda.amp.autocast():
                    logits = model(x); loss = F.cross_entropy(logits,y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                logits = model(x); loss = F.cross_entropy(logits,y); loss.backward(); opt.step()
            accs.append(accuracy(logits.detach(), y))
        tr = float(np.mean(accs))

        # val
        model.eval(); vals=[]
        with torch.no_grad():
            for x,y in tqdm(va_loader, desc=f"val-{tag}", leave=False):
                x,y=x.to(DEVICE), y.to(DEVICE); vals.append(accuracy(model(x),y))
        va = float(np.mean(vals))
        print(f"[{tag}] epoch {ep+1}/{cfg.epochs} train_acc={tr:.4f} val_acc={va:.4f}")
        if va>best: best=va; torch.save({"model":model.state_dict()}, best_path)
        if args.quick: break

    # test on best
    ckpt = torch.load(best_path, map_location=DEVICE); model.load_state_dict(ckpt["model"]); model.eval()
    taccs=[]; correct=0; total=0
    with torch.no_grad():
        for x,y in tqdm(te_loader, desc=f"test-{tag}"):
            x,y=x.to(DEVICE), y.to(DEVICE); p=model(x).argmax(1)
            taccs.append((p==y).float().mean().item()); correct+=(p==y).sum().item(); total+=y.numel()
    acc_clean=float(np.mean(taccs)); N_total=int(correct)
    return model, {"test_acc_clean":acc_clean, "N_total":N_total, "test_total":total}

# ---------- AdaIN helpers ----------
def _denorm01(t):
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(1,3,1,1)
    std  = torch.tensor(IMAGENET_STD,  device=t.device).view(1,3,1,1)
    return (t*std + mean).clamp(0,1)

def _renorm_imagenet(t):
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(1,3,1,1)
    std  = torch.tensor(IMAGENET_STD,  device=t.device).view(1,3,1,1)
    return (t - mean) / std

class StylizedSTLTest(Dataset):
    """
    Stylize each TEST image with a style from a *different* class via official AdaIN.
    Returns (stylized_tensor, shape_label, texture_label, content_idx, style_idx).
    """
    def __init__(self, test_ds: datasets.STL10, alpha=1.0):
        assert test_ds.split=="test"
        self.base = datasets.STL10(root=test_ds.root, split="test", download=False, transform=tf_test())
        self.alpha = alpha
        # index by class
        self.by_cls = {c:[] for c in range(NUM_CLASSES)}
        for i in range(len(self.base)):
            _, y = self.base[i]; self.by_cls[int(y)].append(i)

        # load official AdaIN
        vgg = adain_vgg; vgg.load_state_dict(torch.load(ADAIN_VGG, map_location="cpu"))
        self.vgg = nn.Sequential(*list(vgg.children())[:31]).to(DEVICE).eval()
        dec = adain_decoder; dec.load_state_dict(torch.load(ADAIN_DEC, map_location="cpu"))
        self.dec = dec.to(DEVICE).eval()
        for p in self.vgg.parameters(): p.requires_grad_(False)
        for p in self.dec.parameters(): p.requires_grad_(False)

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        x, y_shape = self.base[i]
        # choose a *different* class for the style
        donor_cls = random.choice([c for c in range(NUM_CLASSES) if c!=y_shape])
        j = random.choice(self.by_cls[donor_cls])
        s, _ = self.base[j]

        # to [0,1] for AdaIN
        c = _denorm01(x.unsqueeze(0).to(DEVICE))
        s = _denorm01(s.unsqueeze(0).to(DEVICE))
        with torch.no_grad():
            cF, sF = self.vgg(c), self.vgg(s)
            tF = adain(cF, sF)
            tF = self.alpha * tF + (1-self.alpha)*cF
            y = self.dec(tF).clamp(0,1)
        y = _renorm_imagenet(y).squeeze(0).cpu()
        return y, y_shape, donor_cls, i, j

class GraySTLTest(Dataset):
    """Grayscale test set (shape/labels unchanged)."""
    def __init__(self, test_ds: datasets.STL10):
        self.base = datasets.STL10(root=test_ds.root, split="test", download=False, transform=tf_gray3())
    def __len__(self): return len(self.base)
    def __getitem__(self, i): return self.base[i]


# --------
def denorm_to_numpy01(t: torch.Tensor) -> np.ndarray:
    """ImageNet-normalized CHW tensor -> HWC float in [0,1]."""
    mean = np.array(IMAGENET_MEAN).reshape(3,1,1)
    std  = np.array(IMAGENET_STD).reshape(3,1,1)
    x = t.detach().cpu().numpy()
    x = np.clip(x*std + mean, 0, 1)
    return np.transpose(x, (1,2,0))

def edge_map(img01: np.ndarray) -> np.ndarray:
    g = rgb2gray(img01)  # [0,1]
    e = canny(g, sigma=1.5)
    return e.astype(np.float32)

def edge_ssim(content01: np.ndarray, styl01: np.ndarray) -> float:
    ec, es = edge_map(content01), edge_map(styl01)
    # SSIM expects [0,1] floats; edges are {0,1}
    return float(ssim(ec, es, data_range=1.0))

def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    # feat: 1xCxHxW
    B,C,H,W = feat.shape
    F = feat.view(C, H*W)
    G = F @ F.t() / (C*H*W)
    return G

def softmax_confidence(logits: torch.Tensor, target_idx: torch.Tensor) -> float:
    # logits: 1xK, target_idx: 1
    p = torch.softmax(logits, dim=1)
    return float(p[0, target_idx.item()].cpu())


# ---------- Metrics ----------
@torch.no_grad()
def eval_clean_gray(model, gray_loader)->Tuple[float,int,int]:
    model.eval(); accs=[]; correct=0; total=0
    for x,y in tqdm(gray_loader, leave=False):
        x,y=x.to(DEVICE), y.to(DEVICE)
        p=model(x).argmax(1)
        accs.append((p==y).float().mean().item()); correct+=(p==y).sum().item(); total+=y.numel()
    return float(np.mean(accs)), int(correct), int(total)

@torch.no_grad()
def eval_stylized_shape_bias(model, styl_loader)->Tuple[int,int,int]:
    """
    Returns (shape_hits, texture_hits, total), where:
      - shape_hits: prediction == shape_label (content class)
      - texture_hits: prediction == texture_label (style donor class)
      - total: number of stylized samples seen
    """
    model.eval(); shape_hits=0; texture_hits=0; total=0
    for batch in tqdm(styl_loader, leave=False):
        x, y_shape, y_tex = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
        p = model(x).argmax(1)
        shape_hits += (p==y_shape).sum().item()
        texture_hits += (p==y_tex).sum().item()
        total += y_shape.numel()
    return shape_hits, texture_hits, total

def select_and_plot_conflicts(styl_ds: StylizedSTLTest, res_model, vit_model, save_path, K=10):
    """
    Build a candidate pool, score each sample, select K with diversity & clear cues,
    and plot tiles with both models' predictions.
    """
    res_model.eval(); vit_model.eval()
    rng = np.random.RandomState(SEED)

    candidates = []
    # iterate test set once (batch=1 for scoring)
    loader = DataLoader(styl_ds, batch_size=1, shuffle=False, num_workers=0)
    with torch.no_grad():
        for (x_styl, y_shape, y_tex, idx_c, idx_s) in tqdm(loader, desc="score-stylized", leave=False):
            # stylized (normalized) -> numpy01 for edges
            vis = denorm_to_numpy01(x_styl[0])
            # original content (normalized)
            x_content, _ = styl_ds.base[int(idx_c)]
            cont_vis = denorm_to_numpy01(x_content)

            # metrics
            e_sim = edge_ssim(cont_vis, vis)  # shape preservation
            # Gram Δ on relu4_1 features (content vs stylized)
            c01 = _denorm01(x_content.unsqueeze(0).to(DEVICE))
            s01 = _denorm01(x_styl.to(DEVICE))
            Gc = gram_matrix(styl_ds.vgg(c01))
            Gs = gram_matrix(styl_ds.vgg(s01))
            gram_delta = float(torch.norm(Gc - Gs, p='fro').cpu())

            # model confidences on shape/texture
            xr = x_styl.to(DEVICE)
            lr = res_model(xr); lv = vit_model(xr)
            conf_shape = max(softmax_confidence(lr, y_shape), softmax_confidence(lv, y_shape))
            conf_tex   = max(softmax_confidence(lr, y_tex),   softmax_confidence(lv, y_tex))
            # does at least one model predict shape/texture?
            pr_r = lr.argmax(1).item(); pr_v = lv.argmax(1).item()
            hit_shape = (pr_r == y_shape.item()) or (pr_v == y_shape.item())
            hit_tex   = (pr_r == y_tex.item())   or (pr_v == y_tex.item())
            informative = hit_shape or hit_tex

            candidates.append({
                "idx": int(idx_c), "style_idx": int(idx_s),
                "y_shape": int(y_shape), "y_tex": int(y_tex),
                "edge_ssim": e_sim, "gram_delta": gram_delta,
                "conf_max": float(max(conf_shape, conf_tex)),
                "pr_res": pr_r, "pr_vit": pr_v,
                "informative": informative,
                "tile": torch.from_numpy(np.transpose(vis, (2,0,1)))  # CHW in [0,1]
            })

    if not candidates:
        print("No candidates found."); return

    # Normalize gram_delta across pool and build composite score
    gvals = np.array([c["gram_delta"] for c in candidates], dtype=np.float32)
    if gvals.std() < 1e-6: gnorm = (gvals - gvals.min())
    else: gnorm = (gvals - gvals.mean()) / (gvals.std() + 1e-6)
    for c, g in zip(candidates, gnorm):
        c["score"] = 0.5*c["edge_ssim"] + 0.3*float(g) + 0.2*c["conf_max"]
        # prefer cases where res/vit disagree on S vs T (diagnostic)
        c["disagree_bonus"] = 0.1 if ( (c["pr_res"]==c["y_tex"]) ^ (c["pr_vit"]==c["y_shape"]) ) else 0.0
        c["score"] += c["disagree_bonus"]

    # Filter: informative + decent shape & style strength
    e_thresh = np.percentile([c["edge_ssim"] for c in candidates], 60)  
    kept = [c for c in candidates if c["informative"] and c["edge_ssim"] >= e_thresh]
    if len(kept) < K: kept = sorted(candidates, key=lambda d: d["score"], reverse=True)

    # Enforce diversity: <= 2 per shape class
    per_class = {i:0 for i in range(NUM_CLASSES)}
    final = []
    for c in sorted(kept, key=lambda d: d["score"], reverse=True):
        if per_class[c["y_shape"]] >= 2: continue
        final.append(c); per_class[c["y_shape"]] += 1
        if len(final) == K: break
    if len(final) < K:
        # fill remaining without class cap
        extras = [c for c in sorted(kept, key=lambda d: d["score"], reverse=True) if c not in final]
        final += extras[:K-len(final)]

    # Plot figure with both predictions under each tile
    tiles = [c["tile"] for c in final]
    grid = make_grid(torch.stack(tiles), nrow=5, padding=2)
    plt.figure(figsize=(16,7))
    plt.imshow(np.transpose(grid.numpy(), (1,2,0))); plt.axis('off')
    plt.title("Stylized cue-conflict examples (auto-selected)")
    # captions
    for idx, c in enumerate(final):
        s = CLASS_NAMES[c["y_shape"]]; t = CLASS_NAMES[c["y_tex"]]
        pr = CLASS_NAMES[c["pr_res"]]; pv = CLASS_NAMES[c["pr_vit"]]
        text = f"S:{s} | T:{t}\nResNet:{pr} | ViT:{pv}"
        row, col = divmod(idx, 5)

        # position label of each tile
        tile_h = grid.shape[1] // (K//5)   # height of each row
        tile_w = grid.shape[2] // 5        # width per column
        x = col*tile_w + 5
        y = row*tile_h + tile_h - 10     
        plt.text(x, y, text, fontsize=8, color='white',
                bbox=dict(facecolor='black', alpha=0.6, pad=1))
    plt.tight_layout(); plt.savefig(save_path, dpi=180); plt.close()


# ---------- Run one model ----------
def run_one(model_name, lr, train_ds, val_ds, test_ds):
    tag = model_name
    model = make_resnet50() if tag=="resnet50" else make_vit_s16()
    cfg = TrainCfg(epochs=args.epochs, batch=args.batch_size, lr=lr, workers=args.workers)
    model, base = train_eval(model, cfg, train_ds, val_ds, test_ds, tag)

    # grayscale (color drop)
    gray_ds = GraySTLTest(test_ds)
    gray_loader = DataLoader(gray_ds, batch_size=cfg.batch, shuffle=False, num_workers=0)
    acc_gray, _, _ = eval_clean_gray(model, gray_loader)
    color_drop_pp = (base["test_acc_clean"] - acc_gray) * 100.0

    # stylized (shape vs texture)
    styl_ds = StylizedSTLTest(test_ds, alpha=args.alpha)
    styl_loader = DataLoader(styl_ds, batch_size=cfg.batch, shuffle=False, num_workers=0)
    shape_hits, texture_hits, total = eval_stylized_shape_bias(model, styl_loader)
    denom = max(1, (shape_hits + texture_hits))
    shape_bias_pct = 100.0 * shape_hits / denom
    coverage_pct = 100.0 * (shape_hits + texture_hits) / total
    summary = {
        "acc_clean": base["test_acc_clean"],
        "acc_gray": acc_gray,
        "color_drop_pp": color_drop_pp,
        "N_total_clean": base["N_total"],
        "styl_total": total,
        "shape_hits": int(shape_hits),
        "texture_hits": int(texture_hits),
        "shape_bias_percent": shape_bias_pct,
        "shape_tex_coverage_percent": coverage_pct
    }
    with open(OUT/f"{tag}_summary.json","w") as f: json.dump(summary, f, indent=2)
    print(f"[{tag}] summary:", json.dumps(summary, indent=2))
    return model, summary, styl_ds

# ---------- Main ----------
def main():
    train_ds, val_ds, test_ds = prepare_stl10(args.data_root)
    print(f"Data: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    res_model, res_sum, styl_ref = run_one("resnet50", args.lr_resnet, train_ds, val_ds, test_ds)
    vit_model, vit_sum, _       = run_one("vit_s16",  args.lr_vit,    train_ds, val_ds, test_ds)

    # combined conflict figure with both predictions
    select_and_plot_conflicts(styl_ref, res_model, vit_model, OUT/"stylized_conflicts_auto.png", K=10)

    # simple results table (png + json)
    table = [
        ["ResNet-50", f"{res_sum['shape_bias_percent']:.2f}", f"{res_sum['shape_tex_coverage_percent']:.1f}",
         f"{res_sum['acc_clean']:.2%}", f"{res_sum['acc_gray']:.2%}", f"{res_sum['color_drop_pp']:.2f}"],
        ["ViT-S/16",  f"{vit_sum['shape_bias_percent']:.2f}", f"{vit_sum['shape_tex_coverage_percent']:.1f}",
         f"{vit_sum['acc_clean']:.2%}", f"{vit_sum['acc_gray']:.2%}", f"{vit_sum['color_drop_pp']:.2f}"],
    ]
    cols = ["Model","Shape bias (%)","Coverage (%)","Clean acc","Gray acc","Color drop (pp)"]
    fig,ax = plt.subplots(figsize=(10,2.6)); ax.axis('off')
    tbl=ax.table(cellText=table, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.25)
    plt.title("STL-10: Stylized Shape-bias (per spec) + Grayscale color drop", pad=10)
    plt.savefig(OUT/"results_table.png", dpi=200, bbox_inches='tight'); plt.close()

    with open(OUT/"combined_report.json","w") as f:
        json.dump({"resnet50":res_sum, "vit_s16":vit_sum}, f, indent=2)

    print("Artifacts written to:", OUT.resolve())

if __name__ == "__main__":
    main()
