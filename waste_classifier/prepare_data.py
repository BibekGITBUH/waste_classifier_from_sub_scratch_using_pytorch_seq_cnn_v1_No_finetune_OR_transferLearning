"""
Splits datasets/final/<class>/*.jpg into:
    datasets/split/train/<class>/*.jpg
    datasets/split/val/<class>/*.jpg
    datasets/split/test/<class>/*.jpg

Run this ONCE before training:
    python prepare_data.py
"""

import random
import shutil
from pathlib import Path

import config


def split_dataset():
    random.seed(config.SPLIT_SEED)
    src_root = Path(config.RAW_DATA_DIR)
    dst_root = Path(config.SPLIT_DATA_DIR)

    if not src_root.exists():
        raise FileNotFoundError(
            f"Could not find {src_root.resolve()}. "
            "Run the dataset download/organize script first."
        )

    for split in ["train", "val", "test"]:
        for cls in config.CLASS_NAMES:
            (dst_root / split / cls).mkdir(parents=True, exist_ok=True)

    summary = {}
    for cls in config.CLASS_NAMES:
        cls_dir = src_root / cls
        if not cls_dir.exists():
            print(f"WARNING: no folder found for class '{cls}' at {cls_dir}, skipping.")
            continue

        images = [
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * config.TRAIN_RATIO)
        n_val = int(n * config.VAL_RATIO)
        # remainder goes to test to avoid rounding loss
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split, files in splits.items():
            for f in files:
                shutil.copy(f, dst_root / split / cls / f.name)

        summary[cls] = {k: len(v) for k, v in splits.items()}

    print("\nDataset split complete:")
    print(f"{'Class':<12}{'Train':<10}{'Val':<10}{'Test':<10}")
    for cls, counts in summary.items():
        print(f"{cls:<12}{counts['train']:<10}{counts['val']:<10}{counts['test']:<10}")

    print(f"\nSplit data written to: {dst_root.resolve()}")


if __name__ == "__main__":
    split_dataset()
