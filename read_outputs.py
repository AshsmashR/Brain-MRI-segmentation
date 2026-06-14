#!/usr/bin/env python3
"""
read_outputs.py
---------------
After FastSurfer finishes, this script opens the key output files and prints a
short, human-readable summary. Its purpose is *understanding* the pipeline:
what FastSurfer produces and where to find it.

It reads only standard FastSurfer/FreeSurfer outputs; it does not modify them.

Usage:
    python scripts/read_outputs.py --subject_dir ~/my_fastsurfer_analysis/subject01

Requires: nibabel, numpy  (pip install nibabel numpy)
"""

import os
import argparse
import numpy as np

try:
    import nibabel as nib
except ImportError:
    raise SystemExit("Please install nibabel:  pip install nibabel numpy")


def summarize_segmentation(subject_dir):
    """The whole-brain segmentation FastSurfer produces from the T1."""
    seg_path = os.path.join(subject_dir, "mri", "aparc.DKTatlas+aseg.deep.mgz")
    if not os.path.exists(seg_path):
        print(f"[seg] not found: {seg_path}")
        return
    seg = nib.load(seg_path)
    data = np.asarray(seg.dataobj)
    labels = np.unique(data)
    voxvol = float(np.prod(seg.header.get_zooms()[:3]))  # mm^3 per voxel
    print("\n=== Whole-brain segmentation (aparc.DKTatlas+aseg.deep) ===")
    print(f" file        : {seg_path}")
    print(f" volume dims : {data.shape}")
    print(f" voxel size  : {seg.header.get_zooms()[:3]} mm")
    print(f" # labels    : {len(labels)} distinct anatomical regions")
    brain_vox = int(np.count_nonzero(data))
    print(f" brain volume: {brain_vox * voxvol / 1000:.1f} cm^3 (non-background)")


def summarize_stats(subject_dir):
    """The numeric stats tables (volumes, thickness) FastSurfer writes."""
    stats_dir = os.path.join(subject_dir, "stats")
    if not os.path.isdir(stats_dir):
        print(f"\n[stats] not found: {stats_dir}")
        return
    print("\n=== Stats tables ===")
    for fname in sorted(os.listdir(stats_dir)):
        if fname.endswith(".stats"):
            print(f"  {os.path.join(stats_dir, fname)}")
    aseg = os.path.join(stats_dir, "aseg.stats")
    if os.path.exists(aseg):
        print("\n  First volume rows from aseg.stats:")
        shown = 0
        with open(aseg) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    # columns: Index SegId NVoxels Volume_mm3 StructName ...
                    print(f"    {parts[4]:<22} {parts[3]:>10} mm^3")
                    shown += 1
                if shown >= 8:
                    print("    ...")
                    break


def summarize_surfaces(subject_dir):
    """The cortical surface meshes — the part most relevant to surface analysis."""
    surf_dir = os.path.join(subject_dir, "surf")
    print("\n=== Cortical surfaces (surf/) ===")
    if not os.path.isdir(surf_dir):
        print(f"  not found: {surf_dir}  (did the surface stream run?)")
        return
    for name in ("lh.white", "rh.white", "lh.pial", "rh.pial",
                 "lh.thickness", "rh.thickness"):
        path = os.path.join(surf_dir, name)
        if os.path.exists(path):
            try:
                if name.endswith(("white", "pial")):
                    verts, faces = nib.freesurfer.read_geometry(path)
                    print(f"  {name:<14} mesh: {len(verts):>7} vertices, {len(faces):>7} faces")
                else:
                    vals = nib.freesurfer.read_morph_data(path)
                    print(f"  {name:<14} thickness: mean {np.mean(vals):.2f} mm "
                          f"over {len(vals)} vertices")
            except Exception as e:
                print(f"  {name:<14} present (could not parse: {e})")
        else:
            print(f"  {name:<14} not present")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject_dir", required=True,
                    help="Path to one subject's FastSurfer output folder")
    args = ap.parse_args()
    sd = args.subject_dir
    if not os.path.isdir(sd):
        raise SystemExit(f"Subject dir not found: {sd}")
    print(f"Reading FastSurfer outputs in: {sd}")
    summarize_segmentation(sd)
    summarize_stats(sd)
    summarize_surfaces(sd)
    print("\nDone. These are FastSurfer's outputs; this script only reads them.")


if __name__ == "__main__":
    main()
