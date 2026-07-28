#!/bin/bash
# exp40 — recalibrate ONE head as a batch, at a chosen PIMC strength (default 32).
# Usage: bash batch_one.sh <head> <N> [PIMC_N]
# Trains + eval at the SAME strong opponent (no train≠test mismatch → no optimism bleeders).
# Overwrites <head>_real_baseline.pt / _isotonic.npz (the tournament loads these).
set -u
cd /Users/milansimity/Cuccok/kodok/oldtawer
PY=.venv/bin/python; DIR=experiments/40_realistic_recal
H=$1; N=$2; PN=${3:-32}
export WORKERS=8 PIMC_N=$PN
stamp(){ date "+%Y-%m-%d %H:%M:%S"; }
echo "[$(stamp)] >>> batch $H  N=$N  PIMC_N=$PN" | tee -a "$DIR/batch.log"
HEAD=$H N=$N SEED_BASE=800000000 OUT="$DIR/${H}_real_p${PN}.npz" \
  $PY "$DIR/datagen.py" > "$DIR/${H}_p${PN}_datagen.log" 2>&1
tail -2 "$DIR/${H}_p${PN}_datagen.log" | sed 's/^/    /' | tee -a "$DIR/batch.log"
HEAD=$H DATA="$DIR/${H}_real_p${PN}.npz" \
  $PY "$DIR/train.py" > "$DIR/${H}_p${PN}_train.log" 2>&1
grep -E "REALISTIC head|GOD head" "$DIR/${H}_p${PN}_train.log" | sed 's/^/    /' | tee -a "$DIR/batch.log"
echo "[$(stamp)] <<< batch $H DONE (PIMC_N=$PN)" | tee -a "$DIR/batch.log"
