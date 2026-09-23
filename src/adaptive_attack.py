"""
Adaptive attack — the project's key differentiator.

Standard evaluations assume the attacker does not know a detector exists. Here
the attacker has full white-box access to BOTH the classifier and the detector,
and optimises a joint objective:

    maximise  CE(classifier(x_adv), y)        # fool the classifier
    minimise  CE(detector(x_adv), "clean")    # evade the detector

Implemented as a PGD variant under an Linf budget.
"""
import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config
from src import data, models


def adaptive_pgd(classifier, detector, x, y,
                 eps=config.EPS, alpha=config.PGD_ALPHA,
                 steps=config.ADAPTIVE_STEPS, lam=config.ADAPTIVE_LAMBDA):
    """Return adversarial images that fool the classifier AND evade the detector."""
    x_orig = x.clone().detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    clean_target = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    for _ in range(steps):
        x_adv.requires_grad_(True)

        loss_cls = F.cross_entropy(classifier(x_adv), y)
        loss_det = F.cross_entropy(detector(x_adv), clean_target)

        # Ascend on classifier loss, descend on detector loss.
        loss = loss_cls - lam * loss_det

        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = x_orig + torch.clamp(x_adv - x_orig, -eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    return x_adv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=config.N_ATTACK_SAMPLES)
    ap.add_argument("--lam", type=float, default=config.ADAPTIVE_LAMBDA)
    args = ap.parse_args()

    device = config.DEVICE
    classifier = models.load_classifier(device=device)
    detector = models.load_detector(device=device)
    _, _, test_loader = data.get_loaders()

    clean_all, adv_all, label_all = [], [], []
    n_done = 0
    fooled = evaded = both = total = 0

    for x, y in tqdm(test_loader, desc="adaptive attack"):
        if n_done >= args.n:
            break
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            mask = classifier(x).argmax(dim=1) == y
        if mask.sum() == 0:
            continue
        x, y = x[mask], y[mask]

        x_adv = adaptive_pgd(classifier, detector, x, y, lam=args.lam)

        with torch.no_grad():
            misclassified = classifier(x_adv).argmax(dim=1) != y
            undetected = detector(x_adv).argmax(dim=1) == 0

        fooled += misclassified.sum().item()
        evaded += undetected.sum().item()
        both += (misclassified & undetected).sum().item()
        total += y.size(0)

        clean_all.append(x.cpu())
        adv_all.append(x_adv.cpu())
        label_all.append(y.cpu())
        n_done += y.size(0)

    out = os.path.join(config.ADV_DIR, "adaptive.pt")
    torch.save({
        "clean": torch.cat(clean_all)[: args.n],
        "adv": torch.cat(adv_all)[: args.n],
        "labels": torch.cat(label_all)[: args.n],
        "stats": {"attack": "adaptive", "n": total},
    }, out)

    print(f"\n--- Adaptive attack results (n={total}, lambda={args.lam}) ---")
    print(f"Classifier fooled:            {fooled / max(total,1):.4f}")
    print(f"Detector evaded:              {evaded / max(total,1):.4f}")
    print(f"BOTH (full attack success):   {both / max(total,1):.4f}")
    print(f"Detector recall under attack: {1 - evaded / max(total,1):.4f}")
    print(f"\nSaved -> {out}")
    print("Compare this detector recall against the non-adaptive number from "
          "src.evaluate. The drop is your headline result.")


if __name__ == "__main__":
    main()
