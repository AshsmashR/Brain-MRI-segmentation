"""
data.py
-------
Loads images + COCO-style polygon annotations and rasterises them into binary
masks, then prepares normalised train/val/test arrays.

Changes vs. the original notebook
==================================
* All paths come from a config dict instead of being hardcoded to E:\\ and
  /kaggle/working — so the same code runs anywhere.
* Mask generation matches each image to its own annotation by image_id
  (the original zipped images and annotations by position, which silently
  breaks if their order ever differs).
* Empty / missing segmentations are skipped gracefully instead of crashing.
"""

import os
import json
import glob
import cv2
import numpy as np


def load_coco_json(folder):
    """Load the first *.json COCO file found in a folder."""
    files = glob.glob(os.path.join(folder, "*.json"))
    if not files:
        raise FileNotFoundError(f"No COCO .json annotation found in {folder}")
    with open(files[0], "r") as fh:
        return json.load(fh)


def generate_masks(image_dir, coco, mask_dir):
    """Rasterise polygon annotations into binary PNG masks (0 / 255)."""
    os.makedirs(mask_dir, exist_ok=True)

    # Map image_id -> list of segmentations (robust to ordering / multiplicity)
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    made = 0
    for img in coco["images"]:
        src = os.path.join(image_dir, img["file_name"])
        image = cv2.imread(src)
        if image is None:
            print(f"  [skip] could not read {img['file_name']}")
            continue

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for ann in anns_by_image.get(img["id"], []):
            seg = ann.get("segmentation")
            if isinstance(seg, list) and len(seg) > 0:
                poly = np.array(seg[0], dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [poly], 255)

        cv2.imwrite(os.path.join(mask_dir, img["file_name"]), mask)
        made += 1
    print(f"  {made} masks written to {mask_dir}")


def _load_split(image_dir, mask_dir, coco, size):
    X, y = [], []
    for img in coco["images"]:
        im = cv2.imread(os.path.join(image_dir, img["file_name"]))
        mk = cv2.imread(os.path.join(mask_dir, img["file_name"]), cv2.IMREAD_GRAYSCALE)
        if im is None or mk is None:
            continue
        X.append(cv2.resize(im, size))
        y.append(cv2.resize(mk, size))

    X = np.asarray(X, dtype="float32") / 255.0
    y = np.expand_dims(np.asarray(y, dtype="float32") / 255.0, -1)
    y = (y > 0.5).astype("float32")
    return X, y


def build_dataset(cfg):
    """Return (X_train, y_train, X_val, y_val, X_test, y_test)."""
    size = tuple(cfg["target_size"])
    out = cfg["mask_root"]

    for split in ("train", "val", "test"):
        coco = load_coco_json(cfg[f"{split}_path"])
        generate_masks(cfg[f"{split}_path"], coco, os.path.join(out, f"{split}_masks"))

    tr = _load_split(cfg["train_path"], os.path.join(out, "train_masks"),
                     load_coco_json(cfg["train_path"]), size)
    va = _load_split(cfg["val_path"], os.path.join(out, "val_masks"),
                     load_coco_json(cfg["val_path"]), size)
    te = _load_split(cfg["test_path"], os.path.join(out, "test_masks"),
                     load_coco_json(cfg["test_path"]), size)
    return (*tr, *va, *te)
