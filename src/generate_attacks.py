"""
Stage 2 — generate adversarial examples against the trained classifier.

All four attacks come from the `torchattacks` library (no custom attack code,
per project scope). Saves a labelled clean/adversarial pool to disk.

    python -m src.generate_attacks
    python -m src.generate_attacks --attacks fgsm pgd
"""
import argparse
import os
import torch
import torchattacks
from tqdm import tqdm

import config
from src import data, models


def build_attack(name, model):
    """Return a configured torchattacks attack object."""
    name = name.lower()
    if name == "fgsm":
        return torchattacks.FGSM(model, eps=config.EPS)
    if name == "pgd":
        return torchattacks.PGD(
            model, eps=config.EPS, alpha=config.PGD_ALPHA,
            steps=config.PGD_STEPS, random_start=True,
        )
    if name == "cw":
        return torchattacks.CW(
            model, c=config.CW_C, kappa=0, steps=config.CW_STEPS, lr=0.01
        )
    if name == "square":
        return torchattacks.Square(
            model, norm="Linf", eps=config.EPS,
            n_queries=config.SQUARE_QUERIES, n_restarts=1,
        )
    raise ValueError(f"Unknown attack: {name}")


@torch.no_grad()
def _accuracy(model, x, y):
    return (model(x).argmax(dim=1) == y).float().mean().item()


def generate(model, loader, attack_name, device, max_samples):
    atk = build_attack(attack_name, model)

    clean_all, adv_all, label_all = [], [], []
    n_done = 0
    clean_correct = adv_correct = total = 0

    for x, y in tqdm(loader, desc=f"attack={attack_name}"):
        if n_done >= max_samples:
            break
        x, y = x.to(device), y.to(device)

        # Only attack images the classifier already gets right — attacking an
        # already-misclassified image tells us nothing.
        with torch.no_grad():
            correct_mask = model(x).argmax(dim=1) == y
        if correct_mask.sum() == 0:
            continue
        x, y = x[correct_mask], y[correct_mask]

        x_adv = atk(x, y)

        clean_correct += _accuracy(model, x, y) * y.size(0)
        adv_correct += _accuracy(model, x_adv, y) * y.size(0)
        total += y.size(0)

        clean_all.append(x.cpu())
        adv_all.append(x_adv.cpu())
        label_all.append(y.cpu())
        n_done += y.size(0)

    clean = torch.cat(clean_all)[:max_samples]
    adv = torch.cat(adv_all)[:max_samples]
    labels = torch.cat(label_all)[:max_samples]

    linf = (adv - clean).abs().amax(dim=(1, 2, 3)).mean().item()
    stats = {
        "attack": attack_name,
        "n": int(total),
        "clean_acc": clean_correct / max(total, 1),
        "adv_acc": adv_correct / max(total, 1),
        "mean_linf": linf,
    }
    return clean, adv, labels, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", nargs="+", default=config.ATTACKS)
    ap.add_argument("--n", type=int, default=config.N_ATTACK_SAMPLES)
    ap.add_argument("--ckpt", default=config.CLASSIFIER_CKPT)
    args = ap.parse_args()

    device = config.DEVICE
    model = models.load_classifier(args.ckpt, device)
    _, _, test_loader = data.get_loaders()

    all_stats = []
    for name in args.attacks:
        clean, adv, labels, stats = generate(
            model, test_loader, name, device, args.n
        )
        out = os.path.join(config.ADV_DIR, f"{name}.pt")
        torch.save(
            {"clean": clean, "adv": adv, "labels": labels, "stats": stats}, out
        )
        all_stats.append(stats)
        print(
            f"  {name}: clean_acc={stats['clean_acc']:.3f} "
            f"adv_acc={stats['adv_acc']:.3f} "
            f"mean_Linf={stats['mean_linf']:.4f} -> {out}"
        )

    print("\nAttack success = drop from clean_acc to adv_acc.")
    print("A working attack should push adv_acc near 0.")


if __name__ == "__main__":
    main()
