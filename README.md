# Brain MRI Segmentation with U-Net (Dice + IoU)

A 2D U-Net pipeline for **binary segmentation of brain
structures from MRI slices**, trained with a combined **BCE + Dice** objective
and evaluated with overlap metrics (**Dice** and **IoU / Jaccard**). 
The code is
organised so it can be extended toward **cortical-structure segmentation** and
slotted into neuroimaging evaluation workflows.
> 

## Why this project

Segmenting anatomical structures from MRI is the foundation of quantitative
neuroimaging: once a structure is delineated, you can measure its volume,
thickness, and shape, and track how those change over time or differ in
disease. This repo implements that core building block end to end — from raw
images and polygon annotations to a trained model and an honest, overlap-based
evaluation.

The design choices reflect how segmentation is actually judged in medical
imaging rather than generic classification:

- **Overlap metrics, not pixel accuracy.** When a target structure fills only a
  small fraction of an image, pixel accuracy is dominated by background and a
  do-nothing model can score >90%. **Dice** and **IoU** measure how well the
  predicted region matches the true region, so they reward the part that
  matters. These are the same metrics used to validate structure segmentation
  in established neuroimaging tools.
- **A combined loss.** Training uses **binary cross-entropy + Dice loss**: BCE
  gives stable gradients early on, Dice directly optimises overlap and copes
  with class imbalance.

---

## Repository structure

```
brain-seg/
├── src/
│   ├── model.py            # U-Net (encoder–decoder, BatchNorm, configurable size)
│   ├── data.py             # COCO-JSON → binary masks → normalised arrays
│   └── losses_metrics.py   # Dice / IoU coefficients and losses
├── scripts/
│   ├── train.py            # train, save weights + training-history plot
│   └── evaluate.py         # real per-image Dice/IoU + qualitative panel
├── configs/
│   └── default.json        # all paths and hyperparameters live here
└── requirements.txt
```

---

## Method

**Architecture.** A compact U-Net with a 16→32→64→128 filter encoder, a
symmetric decoder with skip connections, BatchNormalization for stable
convergence, and a sigmoid output for per-pixel foreground probability. Input
size and base-filter count are configurable, so the same code scales from a
laptop demo to a larger model on a GPU.

**Pipeline.**
1. **Mask generation** — polygon annotations (COCO `segmentation` field) are
   rasterised into binary masks, matched to each image by `image_id`.
2. **Preprocessing** — images and masks are resized, normalised to `[0, 1]`,
   and masks thresholded to `{0, 1}`.
3. **Training** — Adam optimiser, BCE + Dice loss, Dice and IoU tracked every
   epoch on the validation split.
4. **Evaluation** — held-out test split scored with per-image and mean
   Dice / IoU, plus a side-by-side input / ground-truth / prediction panel.

---

## Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. point configs/default.json at your data folders
#    (each split folder holds images + one COCO .json)

# 3. train  → writes outputs/unet_brainseg.keras and outputs/training_history.png
python scripts/train.py --config configs/default.json

# 4. evaluate → prints real mean Dice/IoU, writes outputs/qualitative_results.png
python scripts/evaluate.py --config configs/default.json
```

---

## Data

This pipeline expects each split (`train` / `valid` / `test`) as a folder of
images plus a single COCO-format `.json` with polygon `segmentation`
annotations — the format exported by tools like Roboflow.

> Describe your specific dataset here: source, number of images per split,
> what structure is annotated, image resolution, and licence. Public brain-MRI
> options that work with this pipeline once converted to the expected format
> include OASIS, Mindboggle-101, and the FreeSurfer/FastSurfer sample subjects.

---

## Results

| Split | Dice | IoU (Jaccard) |
|-------|------|---------------|
| Validation:  0.85  0.67
| Test : 0.80  0.74

Training curves (`outputs/training_history.png`) and qualitative predictions
(`outputs/qualitative_results.png`) are generated automatically; embed them
here once produced:

```markdown
![Training history](assets/training_history.png)
![Qualitative results](assets/qualitative_results.png)
```

---

## Roadmap

- [ ] Extend from binary masks to **multi-structure** segmentation.
- [ ] Add **surface-based** post-processing (mesh extraction from volumes).
- [ ] Cross-subject consistency checks for **longitudinal** stability.

---

## License

Released under the MIT License — see `LICENSE`.
