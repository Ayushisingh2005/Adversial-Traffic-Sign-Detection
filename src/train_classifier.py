"""
Stage 1 — train the base traffic sign classifier.

    python -m src.train_classifier
    python -m src.train_classifier --dataset cifar10
"""
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from src import data, models


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=config.DATASET)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS_CLASSIFIER)
    ap.add_argument("--out", default=config.CLASSIFIER_CKPT)
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    device = config.DEVICE
    print(f"Device: {device} | Dataset: {args.dataset}")

    train_loader, val_loader, test_loader = data.get_loaders(args.dataset)
    model = models.build_classifier(
        num_classes=config.NUM_CLASSES[args.dataset], device=device
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR * 3,
        steps_per_epoch=len(train_loader), epochs=args.epochs,
    )

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{running / (pbar.n + 1):.4f}")

        val_acc = evaluate(model, val_loader, device)
        print(f"  val acc: {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), args.out)
            print(f"  saved -> {args.out}")

    model.load_state_dict(torch.load(args.out, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f"\nFINAL clean test accuracy: {test_acc:.4f}")
    print("Objective is >= 0.97 on GTSRB.")


if __name__ == "__main__":
    main()
