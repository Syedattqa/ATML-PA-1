import os, math, random, argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms
from PIL import Image

import timm
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

PACS_CLASSES = ["dog","elephant","giraffe","guitar","horse","house","person"]
DOMAINS = ["photo","art_painting","cartoon","sketch"]  # expected folder names

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ---------------------------- Data (Local only) ----------------------------
def find_images(root):
    """Walk PACS folder and collect (path, label, domain). Accepts lowercase or capitalized domain dirs."""
    root = Path(root)
    items = []
    for dom in DOMAINS:
        droot = root / dom  # lowercase (preferred)
        if not droot.exists():
            # accept capitalized short names
            alt_dom = {"photo":"Photo","art_painting":"Art","cartoon":"Cartoon","sketch":"Sketch"}[dom]
            droot = root / alt_dom
        if not droot.exists():
            continue
        for cls in PACS_CLASSES:
            clsdir = droot / cls
            if not clsdir.exists():
                clsdir2 = droot / cls.capitalize()
                if not clsdir2.exists():
                    continue
                clsdir = clsdir2
            for p in clsdir.rglob("*"):
                if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]:
                    items.append((str(p), PACS_CLASSES.index(cls), dom))
    return items

class PACSLocal(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, y, dom = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y, dom, path

def make_splits_local(all_items, holdout_ratio=0.1):
    """Train on Photo+Art+Cartoon; val is a random holdout from those; test=Sketch."""
    train_items = [(p,y,d) for (p,y,d) in all_items if d in ("photo","art_painting","cartoon")]
    sketch_items = [(p,y,d) for (p,y,d) in all_items if d=="sketch"]
    rng = np.random.default_rng(42)
    by_key = defaultdict(list)
    for it in train_items:
        by_key[(it[2], it[1])].append(it)
    train, val = [], []
    for key, lst in by_key.items():
        n = len(lst); k = max(1, int(math.ceil(holdout_ratio*n)))
        idx = set(rng.choice(n, size=k, replace=False).tolist())
        for j, itm in enumerate(lst):
            (val if j in idx else train).append(itm)
    return train, val, sketch_items

def make_class_balanced_sampler(items):
    labels = [y for (_,y,_) in items]
    counts = Counter(labels)
    weights = [1.0 / counts[y] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

# ---------------------------- Models ----------------------------
def build_model(name, num_classes=7, pretrained=True):
    if name == "resnet50":
        return timm.create_model("resnet50", pretrained=pretrained, num_classes=num_classes)
    if name == "vit_s16":
        return timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=num_classes)
    raise ValueError("Unknown model name")

# ---------------------------- Transforms & Eval ----------------------------
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

@torch.no_grad()
def evaluate(model, loader, device, return_details=False):
    model.eval()
    all_y, all_p, all_s, all_paths = [], [], [], []
    for x, y, dom, paths in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = logits.softmax(dim=1).cpu().numpy()
        preds = probs.argmax(1)
        all_y.extend(y.numpy().tolist())
        all_p.extend(preds.tolist())
        all_s.extend(probs.tolist())
        all_paths.extend(paths)
    acc = (np.array(all_p)==np.array(all_y)).mean()
    macro_f1 = f1_score(all_y, all_p, average="macro")
    cm = confusion_matrix(all_y, all_p, labels=list(range(len(PACS_CLASSES))))
    out = {"acc":float(acc), "macro_f1":float(macro_f1), "cm":cm}
    if return_details:
        out.update({"y_true":all_y,"y_pred":all_p,"probs":all_s,"paths":all_paths})
    return out

def train_one(model, loaders, device, epochs=15, base_lr=3e-4, wd=0.05, amp=True):
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best = {"val_acc":-1, "state":None}
    for ep in range(1, epochs+1):
        model.train()
        pbar = tqdm(loaders["train"], desc=f"Epoch {ep}/{epochs}")
        for x, y, dom, _ in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = nn.CrossEntropyLoss()(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=float(loss.detach().cpu()))
        sched.step()
        val_metrics = evaluate(model, loaders["val"], device)
        if val_metrics["acc"] > best["val_acc"]:
            best = {"val_acc":val_metrics["acc"], "state":{k:v.cpu() for k,v in model.state_dict().items()}}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best["val_acc"]

# ---------------------------- Plotting ----------------------------
def plot_confusion(cm, out_png):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=PACS_CLASSES, yticklabels=PACS_CLASSES,
           ylabel='True', xlabel='Predicted', title="Confusion Matrix (Sketch)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color=("white" if cm[i, j] > thresh else "black"))
    fig.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)

def save_failure_grid(details, out_png, k=25):
    y = np.array(details["y_true"]); p = np.array(details["y_pred"]); probs = np.array(details["probs"])
    wrong = np.where(y!=p)[0]
    if wrong.size == 0: return
    conf = probs[np.arange(len(probs)), p]
    idx = wrong[np.argsort(-conf[wrong])[:k]]
    ncol = 5; nrow = int(math.ceil(len(idx)/ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.4*ncol, 2.4*nrow))
    axs = axs.ravel()
    for j, i in enumerate(idx):
        path = details["paths"][i]
        if not os.path.exists(path): 
            axs[j].axis("off"); continue
        img = Image.open(path).convert("RGB")
        axs[j].imshow(img); axs[j].axis("off")
        axs[j].set_title(f"T:{PACS_CLASSES[y[i]]}\nP:{PACS_CLASSES[p[i]]}", fontsize=8)
    for j in range(len(idx), len(axs)): axs[j].axis("off")
    fig.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

# ---------------------------- Main ----------------------------
def main():
    import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out", type=str, default="./runs_pacs_dg")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip training; load existing checkpoints and only run eval/report.")
    args = ap.parse_args()

    # Reprod & device
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.out)

    # --------- Build datasets/loaders (local folders) ---------
    all_items = find_images(args.data_root)
    if len(all_items) == 0:
        raise SystemExit(f"No images found under {args.data_root}. Expected domain folders "
                         f"[photo|art_painting|cartoon|sketch] or [Photo|Art|Cartoon|Sketch].")

    train_items, val_items, test_items = make_splits_local(all_items, holdout_ratio=0.1)
    dtrain = PACSLocal(train_items, get_transforms(True))
    dval   = PACSLocal(val_items,   get_transforms(False))
    dtest  = PACSLocal(test_items,  get_transforms(False))

    sampler = make_class_balanced_sampler(train_items)
    loaders = {
        "train": DataLoader(dtrain, batch_size=args.batch_size, sampler=sampler,
                            num_workers=4, pin_memory=True),
        "val":   DataLoader(dval,   batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True),
        "test":  DataLoader(dtest,  batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True),
    }

    # --------- Helper for markdown fallback ---------
    def safe_to_markdown(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)  # needs 'tabulate'
        except Exception:
            header = " | ".join(df.columns)
            sep = "-|-".join("-"*len(c) for c in df.columns)
            rows = [" | ".join(str(v) for v in r) for r in df.to_numpy().tolist()]
            return "\n".join([header, sep, *rows])

    # --------- Train / Load / Evaluate ---------
    results = []
    for model_name in ["resnet50", "vit_s16"]:
        tag = model_name
        ckpt_path = os.path.join(args.out, f"{tag}.pt")

        if args.skip_train:
            if not os.path.exists(ckpt_path):
                raise SystemExit(f"--skip-train set but checkpoint not found: {ckpt_path}")
            print(f"[{model_name}] Loading checkpoint:", ckpt_path)
            model = build_model(model_name, num_classes=len(PACS_CLASSES), pretrained=False)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.to(device)
            # Recompute VAL so we can still compute drop
            val_metrics = evaluate(model, loaders["val"], device)
            best_val = float(val_metrics["acc"])
        else:
            model = build_model(model_name, num_classes=len(PACS_CLASSES), pretrained=True)
            model, best_val = train_one(model, loaders, device,
                                        epochs=args.epochs, base_lr=args.lr, amp=True)
            # Save immediately
            torch.save(model.state_dict(), ckpt_path)
            print(f"[{model_name}] Saved checkpoint -> {ckpt_path}")

        # Test on Sketch
        test_metrics = evaluate(model, loaders["test"], device, return_details=True)

        # Artifacts (robust to errors)
        try:
            cm_png = os.path.join(args.out, f"{tag}_cm.png")
            plot_confusion(test_metrics["cm"], cm_png)
        except Exception as e:
            print(f"[WARN] Could not save confusion matrix for {tag}: {e}")
            cm_png = ""

        try:
            fails_png = os.path.join(args.out, f"{tag}_fails.png")
            save_failure_grid(test_metrics, fails_png)
        except Exception as e:
            print(f"[WARN] Could not save failure grid for {tag}: {e}")
            fails_png = ""

        drop = float(best_val) - float(test_metrics["acc"])
        results.append({
            "model": tag,
            "val_acc_mean": round(float(best_val), 4),
            "sketch_acc": round(float(test_metrics["acc"]), 4),
            "macro_f1": round(float(test_metrics["macro_f1"]), 4),
            "acc_drop": round(drop, 4),
            "ckpt": ckpt_path,
            "cm_png": cm_png,
            "fails_png": fails_png
        })

    # --------- Save CSV ---------
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.out, "results.csv")
    df.to_csv(csv_path, index=False)
    print("Saved results:", csv_path)

    # --------- Plots ---------
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(results))
        ax.bar(x-0.15, [r["val_acc_mean"] for r in results], width=0.3, label="Val (train domains)")
        ax.bar(x+0.15, [r["sketch_acc"] for r in results], width=0.3, label="Sketch (test)")
        ax.set_xticks(x); ax.set_xticklabels([r["model"] for r in results])
        ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "acc_bar.png"), dpi=180); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar([r["model"] for r in results], [r["acc_drop"] for r in results])
        ax.set_ylabel("Accuracy Drop (Val − Sketch)")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "drop_bar.png"), dpi=180); plt.close(fig)
    except Exception as e:
        print("[WARN] Plotting failed:", e)

    # --------- Markdown report ---------
    try:
        md = [
            "# PACS Domain-Generalization Report",
            "",
            "- Train domains: **Photo + Art + Cartoon**",
            "- Test domain: **Sketch**",
            "- Models: **ResNet-50** (ImageNet-1k) vs **ViT Small Patch16** (ImageNet-1k)",
            "",
            "## Summary Table",
            safe_to_markdown(df),
            "",
            "## Notes",
            "- *Accuracy drop* = mean validation accuracy on training domains minus Sketch accuracy.",
            "- Check confusion matrices and failure grids for texture-vs-shape failure modes."
        ]
        with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md))
    except Exception as e:
        print("[WARN] Writing report.md failed:", e)

    print("\nDone.")



if __name__ == "__main__":
    main()
