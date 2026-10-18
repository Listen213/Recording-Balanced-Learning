#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/00_migrate_from_final.sh /root/autodl-tmp/RIFT_FEWREC_FINAL_FULL [--delete-old]"
  exit 2
fi
SRC="$(cd "$1" && pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DELETE=0
[[ "${2:-}" == "--delete-old" ]] && DELETE=1
[[ "$SRC" == "$ROOT" ]] && { echo "Refusing to migrate from self."; exit 2; }

mkdir -p "$ROOT/legacy/killed_attempt"

echo "[1/6] Preserve killed-attempt logs/status (NO metrics expected)..."
[[ -f "$SRC/logs/final_test.log" ]] && cp -a "$SRC/logs/final_test.log" "$ROOT/legacy/killed_attempt/" || true
[[ -f "$SRC/runs/final_test/RUNNING.json" ]] && cp -a "$SRC/runs/final_test/RUNNING.json" "$ROOT/legacy/killed_attempt/" || true
[[ -f "$SRC/runs/final_test/PROGRESS.json" ]] && cp -a "$SRC/runs/final_test/PROGRESS.json" "$ROOT/legacy/killed_attempt/" || true

move_dir () {
  local rel="$1"
  local s="$SRC/$rel"; local d="$ROOT/$rel"
  if [[ -e "$d" ]]; then echo "  already present: $rel"; return; fi
  [[ -e "$s" ]] || { echo "MISSING REQUIRED: $s"; exit 3; }
  mkdir -p "$(dirname "$d")"
  mv "$s" "$d"
  echo "  migrated: $rel"
}
move_file () {
  local rel="$1"
  local s="$SRC/$rel"; local d="$ROOT/$rel"
  if [[ -e "$d" ]]; then echo "  already present: $rel"; return; fi
  [[ -e "$s" ]] || { echo "MISSING REQUIRED: $s"; exit 3; }
  mkdir -p "$(dirname "$d")"
  mv "$s" "$d"
  echo "  migrated: $rel"
}

echo "[2/6] Move the ALREADY-FROZEN final split/cache/lock. Do NOT regenerate them."
move_dir "manifests_final"
move_dir "cache/beats_final"
move_dir "final_lock"

echo "[3/6] Move BEATs source/checkpoint..."
move_dir "third_party/beats"
move_file "weights/BEATs_iter3_plus_AS2M.pt"

echo "[4/6] Verify FINAL_CONFIG is identical to the frozen lock..."
python - <<'PY'
from final_uatr.common import sha256_file,load_json
lock=load_json('final_lock/FINAL_LOCK.json')
got=sha256_file('FINAL_CONFIG.json')
want=lock['config_sha256']
print('config sha256:',got)
if got!=want:
    raise SystemExit(f'FINAL_CONFIG HASH MISMATCH\nwant={want}\ngot ={got}')
PY

echo "[5/6] Verify the complete frozen lock in the NEW project..."
python verify_final_lock.py --lock final_lock/FINAL_LOCK.json

if [[ "$DELETE" -eq 1 ]]; then
  echo "[6/6] Delete old project only AFTER verification: $SRC"
  cd "$(dirname "$SRC")"
  rm -rf "$SRC"
else
  echo "[6/6] Keep old project (no --delete-old)."
fi

echo "Migration complete. Fresh evaluation will restart from draw 1/61 with the SAME frozen final protocol."
