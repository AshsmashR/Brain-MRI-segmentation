#!/usr/bin/env bash
# =============================================================================
# run_demo.sh — Run FastSurfer on a single T1-weighted MRI using the official
#               Docker image (deepmi/fastsurfer).
#
# This wrapper does NOT reimplement FastSurfer. It calls the published Docker
# container exactly as documented by the Deep-MI lab, with comments explaining
# what each flag does. The point of the demo is clarity: showing the data in,
# the command, and the outputs out.
#
# Docs followed: https://deep-mi.org/FastSurfer/dev/overview/EXAMPLES.html
# =============================================================================
set -euo pipefail

# ---- Edit these three paths for your machine --------------------------------
T1="$HOME/my_mri_data/T1.nii.gz"                 # your input T1 (.nii / .nii.gz / .mgz)
OUT="$HOME/my_fastsurfer_analysis"               # output directory (SUBJECTS_DIR)
FS_LICENSE="$HOME/fs_license/license.txt"        # free FreeSurfer license file
SID="subject01"                                  # name for this subject's output folder
# -----------------------------------------------------------------------------

# A FreeSurfer license is required ONLY for the surface stream. It is free:
#   https://surfer.nmr.mgh.harvard.edu/registration.html
# The segmentation-only stream (--seg_only) does not need it.

mkdir -p "$OUT"

echo ">> Running FastSurfer on: $T1"
echo ">> Output will be written to: $OUT/$SID"

# --gpus all       : use GPU (drop this line and add '--device cpu' at the end for CPU-only)
# -v host:container: mount input, output and license into the container
# --rm --user ...  : clean up container, write files as the current user
# --t1 / --sd /--sid: FastSurfer flags (input T1, output dir, subject id)
# --3T             : tells FastSurfer the scan is 3 Tesla (atlas tuning)
# --parallel       : process left & right hemispheres in parallel
docker run --gpus all \
  -v "$(dirname "$T1")":/data_in \
  -v "$OUT":/data_out \
  -v "$(dirname "$FS_LICENSE")":/fs_license \
  --rm --user "$(id -u):$(id -g)" \
  deepmi/fastsurfer:latest \
  --fs_license /fs_license/"$(basename "$FS_LICENSE")" \
  --t1 /data_in/"$(basename "$T1")" \
  --sid "$SID" \
  --sd /data_out \
  --3T \
  --parallel \
  --threads 4

echo ">> Done. Explore results in: $OUT/$SID"
echo ">> Try:  python scripts/read_outputs.py --subject_dir $OUT/$SID"
