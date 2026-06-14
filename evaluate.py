"""
evaluate.py
-----------
Load a trained model and compute REAL per-image and mean Dice / IoU on the
test split, plus a qualitative panel of predictions. Run this after train.py;
the numbers it prints are the ones that go in the README results table.

    python scripts/evaluate.py --config configs/default.json
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import build_dataset                       # noqa: E402
from losses_metrics import dice_coef, iou_coef, bce_dice_loss  # noqa: E402


def main(cfg_path):
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    *_, X_te, y_te = build_dataset(cfg)

    model = tf.keras.models.load_model(
        "outputs/unet_brainseg.keras",
        custom_objects={"bce_dice_loss": bce_dice_loss,
                        "dice_coef": dice_coef, "iou_coef": iou_coef},
    )
    preds = (model.predict(X_te) >= 0.5).astype("float32")

    dices, ious = [], []
    for t, p in zip(y_te, preds):
        inter = np.sum(t * p)
        dices.append((2 * inter + 1e-6) / (t.sum() + p.sum() + 1e-6))
        ious.append((inter + 1e-6) / (t.sum() + p.sum() - inter + 1e-6))

    print(f"Mean Dice = {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
    print(f"Mean IoU  = {np.mean(ious):.4f} +/- {np.std(ious):.4f}")

    # Qualitative panel
    os.makedirs("outputs", exist_ok=True)
    n = min(4, len(X_te))
    fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
    for i in range(n):
        ax[i, 0].imshow(X_te[i]); ax[i, 0].set_title("input"); ax[i, 0].axis("off")
        ax[i, 1].imshow(y_te[i].squeeze(), cmap="gray"); ax[i, 1].set_title("ground truth"); ax[i, 1].axis("off")
        ax[i, 2].imshow(preds[i].squeeze(), cmap="gray"); ax[i, 2].set_title("prediction"); ax[i, 2].axis("off")
    fig.tight_layout()
    fig.savefig("outputs/qualitative_results.png", dpi=120)
    print("Saved outputs/qualitative_results.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    main(ap.parse_args().config)
