"""Train the one-handed landmark classifier.

    python training/train_classifier.py \
        --dataset datasets/isl_landmarks.csv \
        --epochs 50 \
        --checkpoint-dir checkpoints/isl \
        --csv-file checkpoints/isl/train_log.csv

Augmentation is automatic — landmark-space transforms are applied to every
training batch, so each epoch the model sees a slightly different version of
every sample.
"""
import argparse
import csv
import sys
from pathlib import Path

# Allow `python training/train_classifier.py ...` (script form) to resolve the
# `training`/`src` packages by putting the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from training.augment import augment_one_hand
from training.check_quality import check
from training.model_one_hand import OneHandClassifier

N_META = 3


# Guarantee enough gradient steps per epoch on small datasets. Without this, the
# default batch size (tuned for the large ASL set) gives only ~3 batches/epoch on
# the ~700-row ISL set -> ~150 steps over 50 epochs -> the model badly underfits
# (~60% val acc). See the ISL ablation in the project notes.
MIN_BATCHES_PER_EPOCH = 16


class LandmarkDataset(Dataset):
    def __init__(self, feats: np.ndarray, labels: np.ndarray, augment: bool,
                 mirror: bool = False):
        self._feats = feats.astype(np.float32)
        self._labels = labels.astype(np.int64)
        self._augment = augment
        self._mirror = mirror
        self._rng = np.random.default_rng(0)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        x = self._feats[idx]
        if self._augment:
            x = augment_one_hand(x, self._rng, mirror=self._mirror)
        return torch.from_numpy(x), int(self._labels[idx])


def _load(dataset: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(dataset, header=None)
    classes = sorted(df.iloc[:, 0].astype(str).unique())
    index = {c: i for i, c in enumerate(classes)}
    labels = df.iloc[:, 0].astype(str).map(index).to_numpy()
    feats = df.iloc[:, N_META:].to_numpy(dtype=np.float32)
    return feats, labels, classes


def _stratified_split(labels: np.ndarray, val_split: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * val_split))
        val_idx.extend(idx[:cut])
        train_idx.extend(idx[cut:])
    return np.array(train_idx), np.array(val_idx)


def _evaluate(model, loader, device, loss_fn) -> tuple[float, float]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            total_loss += loss_fn(out, yb).item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / max(n, 1), correct / max(n, 1)


def train(args) -> None:
    # Fail fast on a bad dataset (Spec 3 §7 / Spec 4 §9).
    if check(str(args.dataset), expected_dim=63) != 0:
        raise SystemExit("Data quality gate failed — fix the dataset and retry.")

    feats, labels, classes = _load(args.dataset)
    train_idx, val_idx = _stratified_split(labels, args.val_split)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}; {len(classes)} classes; "
          f"{len(train_idx)} train / {len(val_idx)} val samples")

    # Shrink the batch on small datasets so there are enough updates per epoch.
    eff_batch = min(args.batch_size, max(8, len(train_idx) // MIN_BATCHES_PER_EPOCH))
    if eff_batch != args.batch_size:
        print(f"Small dataset: batch_size {args.batch_size} -> {eff_batch} "
              f"(~{len(train_idx) // eff_batch} batches/epoch) for adequate "
              f"gradient steps.")

    train_loader = DataLoader(
        LandmarkDataset(feats[train_idx], labels[train_idx], augment=True,
                        mirror=args.mirror),
        batch_size=eff_batch, shuffle=True,
    )
    val_loader = DataLoader(
        LandmarkDataset(feats[val_idx], labels[val_idx], augment=False),
        batch_size=eff_batch,
    )

    model = OneHandClassifier(num_classes=len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.csv_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    with open(log_path, "w", newline="") as logf:
        writer = csv.writer(logf)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                running += loss.item() * len(yb)
            train_loss = running / len(train_idx)
            val_loss, val_acc = _evaluate(model, val_loader, device, loss_fn)
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                             f"{val_acc:.4f}"])
            logf.flush()
            if val_acc >= best_acc:
                best_acc = val_acc
                torch.save(
                    {"model_state": model.state_dict(),
                     "num_classes": len(classes),
                     "classes": classes},
                    ckpt_dir / "best.pt",
                )
    print(f"Best val accuracy: {best_acc:.3f}  ->  {ckpt_dir / 'best.pt'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--csv-file", type=Path, required=True)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=256,
                   help="auto-reduced on small datasets to keep enough steps")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mirror", action="store_true",
                   help="enable mirror-flip augmentation (off by default; see "
                        "training/augment.py)")
    train(p.parse_args())


if __name__ == "__main__":
    main()
