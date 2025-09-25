import argparse, random, math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode, functional as TF
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, pairwise_distances

import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, pairwise_distances

import timm


# -------------------
# CLI
# -------------------
p = argparse.ArgumentParser("Feature Representation Analysis (separate script)")
p.add_argument("--data_root", default="./data")
p.add_argument("--weights_dir", default=r"M:\ATML\PA1\out_stl10_focus")
p.add_argument("--resnet_ckpt", default="resnet50_best.pt")
p.add_argument("--vit_ckpt",    default="vit_s16_best.pt")
p.add_argument("--batch_size", type=int, default=128)
p.add_argument("--workers", type=int, default=0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--cache_dir", default="cache_eval")
p.add_argument("--out_dir", default="out_eval")
# sampling/perturb
p.add_argument("--per_class", type=int, default=100, help="samples per class from clean (total=10*per_class)")
p.add_argument("--shift_pixels", type=int, default=8, help="dx shift for the 'shift' set")
# AdaIN paths
p.add_argument("--adain_vgg", default=r"M:\ATML\PA1\weights\vgg_normalised.pth")
p.add_argument("--adain_dec", default=r"M:\ATML\PA1\weights\decoder.pth")
# t-SNE
p.add_argument("--tsne_perplexity", type=float, default=30)
p.add_argument("--tsne_niter", type=int, default=2000)
# behavior
p.add_argument("--force", action="store_true", help="recompute even if outputs exist")
args = p.parse_args()

# -------------------
# Globals
# -------------------
SEED = args.seed
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IM_SIZE = 224
NUM_CLASSES = 10
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CLASS_NAMES = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']

OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(args.cache_dir); CACHE.mkdir(parents=True, exist_ok=True)
EMB_OUT_DIR = OUT / "embeds"; EMB_OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_OUT_DIR = OUT / "figs";   FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------
# Data helpers
# -------------------
def tf_test():
    return transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def load_test():
    return datasets.STL10(root=args.data_root, split="test", download=False, transform=tf_test())

def denorm(x:torch.Tensor)->torch.Tensor:
    if x.dim()==3:
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3,1,1)
        std  = torch.tensor(IMAGENET_STD,  device=x.device).view(3,1,1)
        return (x*std + mean).clamp(0,1)
    elif x.dim()==4:
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1,3,1,1)
        std  = torch.tensor(IMAGENET_STD,  device=x.device).view(1,3,1,1)
        return (x*std + mean).clamp(0,1)
    raise RuntimeError(f"denorm expects 3D/4D, got {tuple(x.shape)}")

class TransformedSTL(Dataset):
    """
    Uses on-disk cache to avoid recomputation: <cache_dir>/<name>.pt
    Cache: dict{ 'images': float32 Nx3xHxW normalized, 'labels': int64 }
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
                x = build_fn(x)
                if x.dim()==4 and x.size(0)==1: x = x.squeeze(0)
                assert x.dim()==3, f"built sample has shape {tuple(x.shape)}"
                xs.append(x.cpu()); ys.append(int(y))
            self.imgs = torch.stack(xs).float()
            self.labels = torch.tensor(ys, dtype=torch.long)
            torch.save({"images":self.imgs, "labels":self.labels}, self.cache_path)
        self.N = self.labels.numel()
    def __len__(self): return self.N
    def __getitem__(self, i): return self.imgs[i], self.labels[i]

def t_shift(dx:int, dy:int):
    def fn(xn):
        x = denorm(xn)
        pad = (abs(dx), abs(dx), abs(dy), abs(dy))
        x = TF.pad(x, pad, padding_mode="reflect")
        x = TF.affine(x, angle=0, translate=(dx, dy), scale=1.0, shear=[0.0,0.0],
                      interpolation=InterpolationMode.BILINEAR)
        x = TF.center_crop(x, [IM_SIZE, IM_SIZE])
        mean = torch.tensor(IMAGENET_MEAN).view(3,1,1); std=torch.tensor(IMAGENET_STD).view(3,1,1)
        x = (x - mean) / std
        if x.dim()==4 and x.size(0)==1: x = x.squeeze(0)
        assert x.dim()==3, f"t_shift produced {tuple(x.shape)}"
        return x
    return fn

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
        from gitadain import decoder as adain_decoder, vgg as adain_vgg
        from gitadain import adaptive_instance_normalization as adain
        vgg = adain_vgg; vgg.load_state_dict(torch.load(args.adain_vgg, map_location="cpu"))
        self.vgg = nn.Sequential(*list(vgg.children())[:31]).to(DEVICE).eval()
        dec = adain_decoder; dec.load_state_dict(torch.load(args.adain_dec, map_location="cpu"))
        self.dec = dec.to(DEVICE).eval()
        for p in self.vgg.parameters(): p.requires_grad_(False)
        for p in self.dec.parameters(): p.requires_grad_(False)
        self._adain = adain
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y_shape = self.base[i]
        donor_cls = random.choice([c for c in range(NUM_CLASSES) if c!=y_shape])
        j = random.choice(self.by_cls[donor_cls])
        s, _ = self.base[j]
        c = _denorm01(x.unsqueeze(0).to(DEVICE))
        s = _denorm01(s.unsqueeze(0).to(DEVICE))
        with torch.no_grad():
            cF, sF = self.vgg(c), self.vgg(s)
            tF = self._adain(cF, sF)
            tF = self.alpha * tF + (1-self.alpha)*cF
            y = self.dec(tF).clamp(0,1)
        y = _renorm_imagenet(y).squeeze(0).cpu()
        return y, int(y_shape), int(donor_cls), int(i), int(j)

class StylizedWrapper(Dataset):
    """
    Wrap StylizedSTLTest to expose (x,y) for loader + keep aligned metadata arrays.
    """
    def __init__(self, base_test):
        self.sty = StylizedSTLTest(base_test, alpha=1.0)
        self.N = len(self.sty)
        ys=[]; donor=[]; cidx=[]
        for i in range(self.N):
            _, yshape, ydon, idx_content, _ = self.sty[i]
            ys.append(int(yshape)); donor.append(int(ydon)); cidx.append(int(idx_content))
        self.labels = torch.tensor(ys, dtype=torch.long)
        self.donors = np.array(donor, dtype=int)
        self.content_idx = np.array(cidx, dtype=int)
    def __len__(self): return self.N
    def __getitem__(self, i):
        x, yshape, _, _, _ = self.sty[i]
        return x, int(yshape)

# -------------------
# Models & ckpt load
# -------------------
def make_resnet50(): return timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
def make_vit_s16():  return timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=NUM_CLASSES)

def load_model(tag:str, ckpt_path:Path):
    model = make_resnet50() if tag=="resnet50" else make_vit_s16()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(DEVICE).eval()
    return model


def safe_tag(tag: str) -> str:
    """Make model tag safe for filenames (no slashes, spaces, etc.)."""
    s = tag.strip().lower()
    # hard kill path separators first
    s = s.replace('/', '-').replace('\\', '-')
    # collapse any other junk to single hyphens
    import re
    s = re.sub(r'[^a-z0-9]+', '-', s)
    return s.strip('-')

# -------------------
# Feature extraction
# -------------------
def extract_prelogits(model: nn.Module, loader: DataLoader) -> np.ndarray:
    """
    Generic penultimate-embedding extractor that handles both ViT and ResNet in timm.

    - If the model has `forward_head(pre_logits=True)`, we use it (e.g., ViT).
    - Otherwise (e.g., ResNet), we take `forward_features(x)` and global-pool if needed.
      For timm ResNet, `forward_features` already returns pooled (N, C), but we guard anyway.
    """
    embs = []
    model.eval()
    with torch.no_grad():
        for x, *_ in tqdm(loader, leave=False):
            x = x.to(DEVICE)

            feats = model.forward_features(x)

            if hasattr(model, "forward_head"):
                # ViT path (and many others)
                try:
                    pre = model.forward_head(feats, pre_logits=True)  # (N, D)
                except TypeError:
                    # some models don't accept pre_logits kwarg; fall back to feats
                    pre = feats
            else:
                # ResNet path: forward_features returns pooled (N, C) in timm.
                # If we ever get a spatial map, pool it.
                if feats.ndim == 4:
                    # Global average pool to (N, C)
                    pre = feats.mean(dim=(2, 3))
                else:
                    pre = feats

            embs.append(pre.detach().cpu())

    return torch.cat(embs, 0).numpy()


def build_subset_loader(ds, sel_idx, batch_size=128):
    sub = Subset(ds, sel_idx)
    return DataLoader(sub, batch_size=batch_size, shuffle=False,
                      num_workers=args.workers, pin_memory=True)

def sample_balanced_indices(clean_ds: TransformedSTL, per_class:int=100):
    idxs_per_cls = {c: [] for c in range(NUM_CLASSES)}
    for i in range(len(clean_ds)):
        y = int(clean_ds.labels[i].item())
        if len(idxs_per_cls[y]) < per_class:
            idxs_per_cls[y].append(i)
        if all(len(v)==per_class for v in idxs_per_cls.values()):
            break
    for c in range(NUM_CLASSES):
        assert len(idxs_per_cls[c]) == per_class, f"class {c} has {len(idxs_per_cls[c])}"
    sel=[]
    for c in range(NUM_CLASSES):
        sel.extend(idxs_per_cls[c])
    return sel

def tsne_2d(emb: np.ndarray, seed:int=42):
    if emb.shape[1] > 50:
        emb50 = PCA(n_components=50, random_state=seed).fit_transform(emb)
    else:
        emb50 = emb
    ts = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        learning_rate='auto',
        max_iter=args.tsne_niter,   # renamed from n_iter to max_iter
        init='pca',
        random_state=seed,
        metric='cosine'
    )
    return ts.fit_transform(emb50)

def plot_tsne_scatter(df: pd.DataFrame, title: str, save_path: Path,
                      color_col: str = "label", marker_col: str = "perturb"):
    plt.figure(figsize=(10, 8))
    marker_map = {"clean":"o","shift8":"^","stylized":"s"}

    # enforce integer class labels for safe indexing into CLASS_NAMES
    if color_col in df.columns:
        df[color_col] = df[color_col].astype(int)

    classes = sorted(df[color_col].dropna().unique().tolist())
    perts = df[marker_col].unique()

    for pert in perts:
        d = df[df[marker_col] == pert]
        for cls in classes:
            dc = d[d[color_col] == cls]
            if dc.empty:
                continue
            cls_idx = int(cls)
            name = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else str(cls_idx)
            plt.scatter(dc["x"], dc["y"], s=20, marker=marker_map.get(pert, "o"),
                        alpha=0.8, label=f"{pert}-{name}")

    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.title(title); plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()

def plot_tsne_stylized_by_texture(df_stylized: pd.DataFrame, title:str, save_path:Path):
    plt.figure(figsize=(10, 8))
    df_stylized = df_stylized.copy()

    # ensure integer donor classes for indexing
    if "style_cls" in df_stylized.columns:
        # drop NaNs just in case; cast remaining to int
        df_stylized = df_stylized.dropna(subset=["style_cls"])
        df_stylized["style_cls"] = df_stylized["style_cls"].astype(int)

    donors = sorted(df_stylized["style_cls"].unique().tolist())
    for don in donors:
        d = df_stylized[df_stylized["style_cls"] == int(don)]
        name = CLASS_NAMES[int(don)] if 0 <= int(don) < len(CLASS_NAMES) else str(int(don))
        plt.scatter(d["x"], d["y"], s=20, alpha=0.8, label=f"style={name}")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.title(title); plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()

def compute_silhouette_cos(emb: np.ndarray, labels: np.ndarray) -> float:
    return float(silhouette_score(emb, labels, metric='cosine'))

def mean_cosine_distance(A: np.ndarray, B: np.ndarray) -> float:
    d = pairwise_distances(A, B, metric='cosine')
    return float(np.mean(np.diag(d)))

# -------------------
# Main routine
# -------------------
def main():
    print(f"Device: {DEVICE}")

    # base datasets (reuses/creates caches: clean & shift8)
    test_ds = load_test()
    clean_ds  = TransformedSTL(test_ds, "clean", lambda x:x)
    shift8_ds = TransformedSTL(test_ds, f"shift_dx{args.shift_pixels}_dy0", t_shift(args.shift_pixels,0))
    stylized_ds = StylizedWrapper(test_ds)

    # same 1000 content indices from CLEAN
    sel_idx = sample_balanced_indices(clean_ds, per_class=args.per_class)

    # map to stylized dataset indices by content_idx
    content_to_sty = {int(stylized_ds.content_idx[i]): i for i in range(len(stylized_ds))}
    sty_sel_idx = [content_to_sty[i] for i in sel_idx]

    # loaders
    bs = args.batch_size
    clean_loader    = build_subset_loader(clean_ds, sel_idx, batch_size=bs)
    shift8_loader   = build_subset_loader(shift8_ds, sel_idx, batch_size=bs)
    stylized_loader = build_subset_loader(stylized_ds, sty_sel_idx, batch_size=bs)

    # models
    res_model = load_model("resnet50", Path(args.weights_dir)/args.resnet_ckpt)
    vit_model = load_model("vit_s16",  Path(args.weights_dir)/args.vit_ckpt)

    def run_model(tag:str, model: nn.Module):
        # ---- safe tag for filenames (no slashes/spaces etc.) ----
        def _safe_tag(t: str) -> str:
            s = t.strip().lower().replace('/', '-').replace('\\', '-')
            import re
            s = re.sub(r'[^a-z0-9]+', '-', s)
            return s.strip('-')
        tag_safe = _safe_tag(tag)
        # ---------------------------------------------------------

        csv_path = EMB_OUT_DIR / f"{tag_safe}_embeddings_clean_shift8_stylized.csv"

        if csv_path.exists() and not args.force:
            print(f"[{tag}] CSV exists, skipping extraction (use --force to recompute): {csv_path}")
            df_all = pd.read_csv(csv_path)

        else:
            print(f"[{tag}] extracting embeddings...")
            emb_clean  = extract_prelogits(model, clean_loader)
            emb_shift8 = extract_prelogits(model, shift8_loader)
            emb_style  = extract_prelogits(model, stylized_loader)

            y_clean  = clean_ds.labels[sel_idx].numpy()
            y_shift8 = shift8_ds.labels[sel_idx].numpy()
            y_style  = stylized_ds.labels[sty_sel_idx].numpy()
            donors   = stylized_ds.donors[sty_sel_idx]
            contents = stylized_ds.content_idx[sty_sel_idx]

            # save CSV
            def to_df(emb, labels, perturb, extra=None):
                df = pd.DataFrame(emb)
                df.insert(0,"label",labels)
                df.insert(1,"perturb",perturb)
                if extra:
                    for k,v in extra.items(): df[k]=v
                return df
            df_all = pd.concat([
                to_df(emb_clean,  y_clean,  "clean"),
                to_df(emb_shift8, y_shift8, "shift8"),
                to_df(emb_style,  y_style,  "stylized", extra={"style_cls": donors, "content_idx": contents})
            ], ignore_index=True)

            # ensure parent exists and save
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_all.to_csv(csv_path, index=False)

            print(f"[{tag}] saved -> {csv_path}")

        # --- Normalize dtypes for safe downstream indexing ---
        if "label" in df_all.columns:
            df_all["label"] = df_all["label"].astype(int)
        if "style_cls" in df_all.columns:
            mask_style = (df_all["perturb"] == "stylized")
            # cast only stylized rows; leave others (NaN) as-is
            df_all.loc[mask_style, "style_cls"] = df_all.loc[mask_style, "style_cls"].astype(int)

        # t-SNE plot (color=class, marker=perturb)
        z_path = FIG_OUT_DIR / f"tsne_{tag_safe}_clean_shift8_stylized.png"
        if not z_path.exists() or args.force:
            feats = df_all.drop(columns=["label","perturb","style_cls","content_idx"], errors='ignore').values
            z = tsne_2d(feats, seed=SEED)
            df_plot = df_all[["label","perturb","style_cls"]].copy()
            df_plot["x"] = z[:,0]; df_plot["y"] = z[:,1]
            # ensure figure directory exists before save inside plot fn
            z_path.parent.mkdir(parents=True, exist_ok=True)
            plot_tsne_scatter(df_plot, f"{tag}: t-SNE (color=class, marker=perturb)", z_path)
            print(f"[{tag}] saved plot -> {z_path}")
        else:
            print(f"[{tag}] plot exists, skipping: {z_path}")

        # stylized-only plot (color=texture donor)
        z2_path = FIG_OUT_DIR / f"tsne_{tag_safe}_stylized_by_texture.png"
        if not z2_path.exists() or args.force:
            feats_style = df_all[df_all["perturb"]=="stylized"].drop(
                columns=["label","perturb","style_cls","content_idx"], errors='ignore'
            ).values
            z_style = tsne_2d(feats_style, seed=SEED)
            df_style_plot = df_all[df_all["perturb"]=="stylized"][["style_cls"]].copy()
            df_style_plot["x"] = z_style[:,0]; df_style_plot["y"] = z_style[:,1]
            # ensure figure directory exists before save inside plot fn
            z2_path.parent.mkdir(parents=True, exist_ok=True)
            plot_tsne_stylized_by_texture(df_style_plot, f"{tag}: Stylized only (color=texture donor)", z2_path)
            print(f"[{tag}] saved plot -> {z2_path}")
        else:
            print(f"[{tag}] plot exists, skipping: {z2_path}")

        # metrics (always recompute quickly from CSV we already have)
        emb_all = df_all.drop(columns=["label","perturb","style_cls","content_idx"], errors='ignore').values
        y_all   = df_all["label"].values
        mask_c  = (df_all["perturb"]=="clean").values
        mask_s8 = (df_all["perturb"]=="shift8").values
        mask_st = (df_all["perturb"]=="stylized").values

        emb_clean = emb_all[mask_c]; y_clean = y_all[mask_c]
        emb_shift8 = emb_all[mask_s8]; y_shift8 = y_all[mask_s8]
        emb_style = emb_all[mask_st]; y_style = y_all[mask_st]

        sil_clean  = compute_silhouette_cos(emb_clean,  y_clean)
        sil_shift8 = compute_silhouette_cos(emb_shift8, y_shift8)
        sil_style  = compute_silhouette_cos(emb_style,  y_style)

        # pairwise drift assumes rows aligned by sampling order; we constructed that way
        drift_shift = mean_cosine_distance(emb_clean, emb_shift8)
        drift_style = mean_cosine_distance(emb_clean, emb_style)

        return {
            "Model": tag,
            "Silhouette_Clean": round(sil_clean,4),
            "Silhouette_Shift8": round(sil_shift8,4),
            "Silhouette_Stylized": round(sil_style,4),
            "Drift_Clean_to_Shift8(cos)": round(drift_shift,4),
            "Drift_Clean_to_Stylized(cos)": round(drift_style,4),
            "N": emb_clean.shape[0]
        }

    summaries = []
    summaries.append(run_model("ResNet-50", res_model))
    summaries.append(run_model("ViT-S/16",  vit_model))

    df_sum = pd.DataFrame(summaries)
    sum_csv = OUT / "feature_space_metrics.csv"
    df_sum.to_csv(sum_csv, index=False)
    print("\nSaved summary ->", sum_csv)
    print(df_sum)

if __name__ == "__main__":
    main()
