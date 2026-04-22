#!/bin/bash
# One-time prerequisite: build Singularity .sif images from the parquet.
# Invocation: s0_build_sifs.sh <start_idx> <end_idx>
set -eo pipefail
START="${1:-1}"
END="${2:-50}"
source /home/ubuntu/.prorl_creds.env
# pull_swe_images.py calls subprocess "python -m openhands.runtime..."; make sure
# "python" resolves to the DLAMI /opt/pytorch env which has openhands + deps.
export PATH="/opt/pytorch/bin:$PATH"
REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
DEST="$REPO/singularity_images"
TEMP="$DEST/temp_dif"
PARQUET=/home/ubuntu/data/SkyRL-v0-293/train.parquet
# Wait for parquet file to appear (the pull runs in parallel).
for i in {1..300}; do
  [ -f "$PARQUET" ] && break
  echo "[sif-build] waiting for $PARQUET ... ($i)"
  sleep 5
done
if [ ! -f "$PARQUET" ]; then
  # Fallback: try top-level filename.
  for candidate in /home/ubuntu/data/SkyRL-v0-293/**/*.parquet; do
    if [ -f "$candidate" ]; then PARQUET="$candidate"; break; fi
  done
fi
echo "[sif-build] parquet: $PARQUET"
echo "[sif-build] dest:    $DEST"
echo "[sif-build] range:   $START..$END"
cd "$REPO"
/opt/pytorch/bin/python3 scripts/pull_swe_images.py \
  --parquet-file "$PARQUET" \
  --dest-dir "$DEST" \
  --temp-base "$TEMP" \
  --start-index "$START" \
  --end-index "$END" \
  --log-name "s0_sif_${START}_${END}"
echo "[sif-build] done $(date -u +%FT%TZ)"
ls "$DEST"/*.sif 2>/dev/null | wc -l | xargs -I{} echo "[sif-build] total .sif: {}"
