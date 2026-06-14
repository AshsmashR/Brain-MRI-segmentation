"""
train.py
--------
Train the U-Net with a combined BCE + Dice objective, tracking Dice and IoU.

Usage
-----
    python scripts/train.py --config configs/default.json

Edit configs/default.json so the paths point at your data, then run. The
script saves the trained weights and a training-history plot, and prints
final validation Dice / IoU.
"""

import os
import sys
import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import build_dataset            # noqa: E402
from model import unet_model              # noqa: E402
from losses_metrics import bce_dice_loss, dice_coef, iou_coef  # noqa: E402


def main(cfg_path):
    with open(cfg_path) as fh:
        cfg = json.load(fh)

    X_tr, y_tr, X_va, y_va, X_te, y_te = build_dataset(cfg)
    print(f"train={X_tr.shape}  val={X_va.shape}  test={X_te.shape}")

    model = unet_model(input_size=(*cfg["target_size"], 3),
                       base_filters=cfg.get("base_filters", 16))
    model.compile(optimizer="adam", loss=bce_dice_loss,
                  metrics=[dice_coef, iou_coef])
    model.summary()

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        batch_size=cfg.get("batch_size", 8),
        epochs=cfg.get("epochs", 10),
    )

    os.makedirs("outputs", exist_ok=True)
    model.save("outputs/unet_brainseg.keras")

    # Training curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history.history["dice_coef"], label="train Dice")
    ax[1].plot(history.history["val_dice_coef"], label="val Dice")
    ax[1].plot(history.history["val_iou_coef"], label="val IoU")
    ax[1].set_title("Overlap metrics"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig("outputs/training_history.png", dpi=120)
    print("Saved outputs/training_history.png")

    # Final held-out evaluation
    test_loss, test_dice, test_iou = model.evaluate(X_te, y_te, verbose=0)
    print(f"\nTEST  loss={test_loss:.4f}  Dice={test_dice:.4f}  IoU={test_iou:.4f}")
    print("Paste these real numbers into the README results table.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    main(ap.parse_args().config)
