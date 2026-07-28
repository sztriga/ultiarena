#!/bin/bash
# exp40 — realistic-recalibration GP gate. RECAL (6 realistic heads) vs FRONTIER (god heads),
# identical engine, played at DEPLOYMENT strength PIMC_N=16 (datagen was 8 → this is the honest
# validation: calibrate cheap, validate at the real defense). Resumable; report() on partial data.
set -u
cd /Users/milansimity/Cuccok/kodok/oldtawer
PY=.venv/bin/python
DIR=experiments/40_realistic_recal
LOG=$DIR/tournament.log
export WORKERS=8 PIMC_N="${PIMC_N:-32}"    # matched to the retrained heads' opponent
stamp(){ date "+%Y-%m-%d %H:%M:%S"; }

run(){
  local M=$1 N=$2 f
  f=$DIR/tour_$(echo "$M" | tr ':' '_').log
  echo "[$(stamp)] >>> $M  N=$N" | tee -a "$LOG"
  MATCHUP=$M N=$N $PY $DIR/tournament.py build > "$f" 2>&1
  tail -1 "$f" | sed 's/^/    /' | tee -a "$LOG"
  echo "[$(stamp)] <<< $M done" | tee -a "$LOG"
}

echo "===== exp40 TOURNAMENT START $(stamp)  PIMC_N=$PIMC_N =====" | tee -a "$LOG"
run "h2h:RECAL:FRONTIER" 3000     # the GATE first (most important) — 3 seatings/deal
run "self:RECAL"          5000    # contract mix + bleeders (per-contract mean/SE)
run "self:FRONTIER"       5000    # baseline mix + bleeders
$PY $DIR/tournament.py report >> "$LOG" 2>&1
echo "===== exp40 TOURNAMENT DONE $(stamp) — see TOURNAMENT.md =====" | tee -a "$LOG"
