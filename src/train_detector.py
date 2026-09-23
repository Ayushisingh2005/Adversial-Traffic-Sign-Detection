"""
Stage 3 — train the adversarial detector (the project's core contribution).

Builds a balanced clean/adversarial dataset from every attack type so the
detector cannot simply memorise one attack signature.

    python -m src.train_detector
    python -m src.train_detector --holdout cw     # leave-one-attack-out test
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

import config
from src import models


def load_pool(attacks, holdout=None):
    """Load saved attack files and return (X, y) where y: 0=clean, 1=adversarial."""
    clean_parts, adv_parts = [], []

    for name in attacks:
        if holdout and name == holdout:
            continue
        path = os.path.join(config.ADV_DIR, f"{name}.pt")
        if not os.path.exists(path):
            print(f"  ! missing {path}, skipping")
            continue
        d = torch.load(path, map_location="cpu")
        clean_parts.append(d["clean"])
        adv_parts.append(d["adv"])
        print(f"  loaded {name}: {d['adv'].shape[0]} adversarial samples")

    if not adv_parts:
        raise RuntimeError("No attack files found. Run src.generate_attacks first.")

    adv = torch.cat(adv_parts)
    clean = torch.cat(clean_parts)

    # Balance the classes: clean images repeat across attack files, so dedupe
    # by subsampling clean down to the adversarial count.
    n = min(len(clean), len(adv))
    perm_c = torch.randperm(len(clean))[:n]
    perm_a = torch.randperm(len(adv))[:n]

    X = torch.cat([clean[perm_c], adv[perm_a]])
    y = torch.cat([torch.zeros(n, dtype=torch.long),
                   torch.ones(n, dtype=torch.long)])
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", nargs="+", default=config.ATTACKS)
    ap.add_argument("--holdout", default=None,
                    help="attack to EXCLUDE from training (generalisation test)")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS_DETECTOR)
    ap.add_argument("--out", default=config.DETECTOR_CKPT)
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    device = config.DEVICE

    print("Building detector training pool...")
    X, y = load_pool(args.attacks, args.holdout)
    print(f"Total samples: {len(X)} (balanced clean/adversarial)")

    ds = TensorDataset(X, y)
    n_val = int(0.15 * len(ds))
    g = torch.Generator().manual_seed(config.SEED)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    detector = models.build_detector(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        detector.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        detector.train()
        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(detector(xb), yb)
            loss.backward()
            optimizer.step()

        detector.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (detector(xb).argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
        acc = correct / max(total, 1)
        print(f"  detector val acc: {acc:.4f}")
        if acc > best:
            best = acc
            torch.save(detector.state_dict(), args.out)
            print(f"  saved -> {args.out}")

    print(f"\nBest detector validation accuracy: {best:.4f}")
    print("Objective is >= 0.90.")
    if args.holdout:
        print(f"NOTE: trained WITHOUT '{args.holdout}'. Evaluate on it to test "
              f"cross-attack generalisation.")


if __name__ == "__main__":
    main()
