#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${1:-/root/autodl-tmp/RIFT_FEWREC_FINAL_FAST_FULL}"

if [[ ! -d "$SRC" ]]; then
  echo "ShipsEar source project not found: $SRC"
  exit 2
fi

echo "[1/4] Copy BEATs source as physical files (dereference symlinks)..."
mkdir -p third_party weights

if [[ -d third_party/beats ]]; then
  echo "third_party/beats already exists; skip."
else
  if [[ ! -d "$SRC/third_party/beats" ]]; then
    echo "Missing $SRC/third_party/beats"
    exit 2
  fi
  cp -aL "$SRC/third_party/beats" third_party/beats
fi

echo "[2/4] Copy BEATs checkpoint as an independent file..."
if [[ -f weights/BEATs_iter3_plus_AS2M.pt ]]; then
  echo "checkpoint already exists; skip."
elif [[ -f "$SRC/weights/BEATs_iter3_plus_AS2M.pt" ]]; then
  cp -aL "$SRC/weights/BEATs_iter3_plus_AS2M.pt" weights/BEATs_iter3_plus_AS2M.pt
elif [[ -f /root/autodl-tmp/weights/BEATs_iter3_plus_AS2M.pt ]]; then
  cp -aL /root/autodl-tmp/weights/BEATs_iter3_plus_AS2M.pt weights/BEATs_iter3_plus_AS2M.pt
else
  echo "Could not find BEATs_iter3_plus_AS2M.pt"
  exit 2
fi

echo "[3/4] Verify this DeepShip project does not depend on symlinks..."
if find third_party/beats weights -type l | grep -q .; then
  echo "ERROR: symlink found after copy:"
  find third_party/beats weights -type l
  exit 2
fi

echo "[4/4] Environment check..."
python check_env.py

echo
echo "Physical copy complete."
echo "ShipsEar project remains untouched: $SRC"
