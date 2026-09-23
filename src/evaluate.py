"""
Evaluation — produces the results tables for the paper.

Computes accuracy / precision / recall / F1 / ROC-AUC for:
  * the proposed CNN detector, per attack and in aggregate
  * the Feature Squeezing baseline, same protocol
  * both under the adaptive threat model (if adaptive.pt exists)

    python -m src.evaluate
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

import config
from src import models, baseline_squeeze


def metrics(y_true, y_pred, y_score=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        out["roc_auc"] = float("nan")
    return out


@torch.no_grad()
def detector_eval(detector, clean, adv, device, batch=256):
    """Return (y_true, y_pred, y_score) for the proposed detector."""
    xs = torch.cat([clean, adv])
    y_true = np.concatenate([np.zeros(len(clean)), np.ones(len(adv))])

    preds, scores = [], []
    for i in range(0, len(xs), batch):
        xb = xs[i:i + batch].to(device)
        logits = detector(xb)
        prob_adv = F.softmax(logits, dim=1)[:, 1]
        preds.append(logits.argmax(dim=1).cpu().numpy())
        scores.append(prob_adv.cpu().numpy())

    return y_true, np.concatenate(preds), np.concatenate(scores)


def squeeze_eval(classifier, clean, adv, device, batch=256):
    """Return (y_true, y_pred, y_score) for the Feature Squeezing baseline."""
    def batched(x):
        out = []
        for i in range(0, len(x), batch):
            out.append(baseline_squeeze.squeeze_scores(
                classifier, x[i:i + batch], device))
        return np.concatenate(out)

    s_clean = batched(clean)
    s_adv = batched(adv)

    thr = baseline_squeeze.pick_threshold(s_clean)
    y_true = np.concatenate([np.zeros(len(s_clean)), np.ones(len(s_adv))])
    y_score = np.concatenate([s_clean, s_adv])
    y_pred = baseline_squeeze.predict(y_score, thr)
    return y_true, y_pred, y_score


def main():
    device = config.DEVICE
    classifier = models.load_classifier(device=device)
    detector = models.load_detector(device=device)

    attack_files = config.ATTACKS + ["adaptive"]
    rows = []

    for name in attack_files:
        path = os.path.join(config.ADV_DIR, f"{name}.pt")
        if not os.path.exists(path):
            continue
        d = torch.load(path, map_location="cpu")
        clean, adv = d["clean"], d["adv"]

        yt, yp, ys = detector_eval(detector, clean, adv, device)
        r = metrics(yt, yp, ys)
        r.update({"attack": name, "method": "Proposed CNN detector"})
        rows.append(r)

        yt, yp, ys = squeeze_eval(classifier, clean, adv, device)
        r = metrics(yt, yp, ys)
        r.update({"attack": name, "method": "Feature Squeezing"})
        rows.append(r)

        print(f"evaluated: {name}")

    if not rows:
        print("No attack files found. Run src.generate_attacks first.")
        return

    df = pd.DataFrame(rows)[
        ["method", "attack", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    out_csv = os.path.join(config.RESULTS_DIR, "detection_results.csv")
    df.to_csv(out_csv, index=False)

    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)
    print(f"\nSaved -> {out_csv}")

    # Headline comparison: non-adaptive vs adaptive for the proposed detector.
    prop = df[df.method == "Proposed CNN detector"]
    std = prop[prop.attack.isin(config.ATTACKS)]["recall"].mean()
    adp = prop[prop.attack == "adaptive"]["recall"]
    if len(adp):
        print(f"\nMean recall, standard attacks : {std:.4f}")
        print(f"Recall under adaptive attack  : {adp.values[0]:.4f}")
        print(f"Robustness drop               : {std - adp.values[0]:.4f}")


if __name__ == "__main__":
    main()
