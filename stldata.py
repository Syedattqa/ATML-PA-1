# file: fetch_stl10.py
from __future__ import annotations

import os
import sys
import tarfile
import errno
import argparse
import urllib.request
from pathlib import Path

import numpy as np

try:
    from imageio import imwrite as imsave  # only used if --save_pngs
except Exception:
    imsave = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# --------------------------
# Constants / expected files
# --------------------------
DATA_URL = "https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
BIN_DIR_NAME = "stl10_binary"
EXPECTED_BIN_FILES = [
    "train_X.bin", "train_y.bin",
    "test_X.bin",  "test_y.bin",
    "unlabeled_X.bin",
    "class_names.txt", "fold_indices.txt", "readme.txt"
]

HEIGHT, WIDTH, DEPTH = 96, 96, 3
IMAGE_SIZE = HEIGHT * WIDTH * DEPTH


# --------------------------
# Helpers
# --------------------------
def human(n: int) -> str:
    for unit in ("B","KB","MB","GB","TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def download_with_progress(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _basic():
        urllib.request.urlretrieve(url, dest)  # fallback, no progress bar

    if tqdm is None:
        print(f"Downloading (no tqdm)… {url}")
        return _basic()

    # With progress bar
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(1024 * 64)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
        pbar.close()


def safe_extract(tar: tarfile.TarFile, path: Path):
    """Prevent path traversal; adapted from Python docs."""
    path = path.resolve()
    for member in tar.getmembers():
        member_path = (path / member.name).resolve()
        if not str(member_path).startswith(str(path)):
            raise Exception("Unsafe path in tar file: " + member.name)
    tar.extractall(path)


def binaries_present(bin_dir: Path) -> bool:
    return all((bin_dir / f).exists() for f in EXPECTED_BIN_FILES)


def ensure_download_and_extract(data_root: Path, force_reextract: bool) -> Path:
    data_root = data_root.resolve()
    archive = data_root / "stl10_binary.tar.gz"
    bin_dir = data_root / BIN_DIR_NAME

    # 1) Download if archive missing
    if not archive.exists():
        print(f"[fetch] Archive missing. Will download to: {archive}")
        download_with_progress(DATA_URL, archive)
    else:
        print(f"[fetch] Archive already present: {archive} ({human(archive.stat().st_size)})")

    # 2) Extract if forced OR bin_dir missing essential files
    need_extract = force_reextract or not binaries_present(bin_dir)
    if need_extract:
        print(f"[extract] Extracting into: {data_root}  (force={force_reextract})")
        with tarfile.open(archive, "r:gz") as tar:
            safe_extract(tar, data_root)
    else:
        print(f"[extract] Skipping extraction; binaries already complete at: {bin_dir}")

    # 3) Verify post-conditions
    if not binaries_present(bin_dir):
        raise FileNotFoundError(
            f"[verify] Missing expected files in {bin_dir}. "
            f"If the previous run was interrupted, try --force_reextract."
        )

    print(f"[verify] All expected files present in: {bin_dir}")
    return bin_dir


# --------------------------
# Reading helpers (optional)
# --------------------------
def read_labels(path_to_labels: Path) -> np.ndarray:
    with open(path_to_labels, "rb") as f:
        return np.fromfile(f, dtype=np.uint8)  # labels are 1..10


def read_all_images(path_to_data: Path) -> np.ndarray:
    with open(path_to_data, "rb") as f:
        everything = np.fromfile(f, dtype=np.uint8)
    images = everything.reshape(-1, 3, HEIGHT, WIDTH)            # N x C x H x W
    images = np.transpose(images, (0, 2, 3, 1))                   # N x H x W x C  (for visualization)
    return images


def save_some_pngs(images: np.ndarray, labels: np.ndarray, out_root: Path, max_per_class: int = 50):
    if imsave is None:
        print("[png] imageio not installed; skipping PNG export.")
        return
    print(f"[png] Exporting up to {max_per_class} images per class → {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    count_per_class = {i: 0 for i in range(1, 11)}  # STL10 uses labels 1..10
    for idx, (img, lab) in enumerate(zip(images, labels)):
        if count_per_class[lab] >= max_per_class:
            continue
        cls_dir = out_root / f"{lab}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        imsave(cls_dir / f"{idx:05d}.png", img)
        count_per_class[lab] += 1
    print("[png] Done.")


# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Safe STL-10 fetcher")
    ap.add_argument("--data_root", type=str, default="./data",
                    help="Root folder where stl10_binary/ will live (default: ./data)")
    ap.add_argument("--force_reextract", action="store_true",
                    help="Force re-extraction even if files already exist")
    ap.add_argument("--save_pngs", action="store_true",
                    help="(Optional) Export a *subset* of training images as PNGs for inspection")
    ap.add_argument("--max_per_class", type=int, default=50,
                    help="Max PNGs per class if --save_pngs is used (default: 50)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    print(f"\n== STL-10 downloader ==")
    print(f"data_root        : {data_root}")
    print(f"force_reextract  : {args.force_reextract}")
    print(f"save_pngs        : {args.save_pngs} (max_per_class={args.max_per_class})\n")

    bin_dir = ensure_download_and_extract(data_root, args.force_reextract)

    # Optional: quick integrity read
    train_X = bin_dir / "train_X.bin"
    train_y = bin_dir / "train_y.bin"

    try:
        labels = read_labels(train_y)
        images = read_all_images(train_X)
        print(f"[read] train images shape: {images.shape}  | labels: {labels.shape} (1..10)")
    except Exception as e:
        print("[read] Warning: could not read binaries:", e)

    if args.save_pngs:
        save_some_pngs(images, labels, out_root=data_root / "stl10_preview_pngs",
                       max_per_class=args.max_per_class)

    print("\n✅ Ready to use with torchvision:")
    print(f"   from torchvision import datasets, transforms")
    print(f"   ds = datasets.STL10(root='{data_root.as_posix()}', split='train', download=False)")
    print(f"   # (Your training script will use this path and apply its own transforms.)\n")


if __name__ == "__main__":
    main()
