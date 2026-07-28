#!/bin/bash
# exp40 — PIMC-32 matched recalibration, end to end (hands-off, runs over ~a day):
#   1. retrain ulti + duri_colored at PIMC-32 (colorless_duri already done)
#   2. clear stale PIMC-16 eval data
#   3. full-ladder GP gate at PIMC-32 (RECAL = 3 retrained heads vs FRONTIER = god) → TOURNAMENT.md
# Bleeders should be GONE now that train opponent == eval opponent (both PIMC-32).
set -u
cd /Users/milansimity/Cuccok/kodok/oldtawer
DIR=experiments/40_realistic_recal
LOG=$DIR/pimc32_master.log
stamp(){ date "+%Y-%m-%d %H:%M:%S"; }
echo "===== PIMC-32 full recal START $(stamp) =====" | tee -a "$LOG"

echo "[$(stamp)] batch 2/3: ulti" | tee -a "$LOG"
bash "$DIR/batch_one.sh" ulti 20000 32
echo "[$(stamp)] batch 3/3: duri_colored (~10h)" | tee -a "$LOG"
bash "$DIR/batch_one.sh" duri_colored 10000 32

# verify all 3 retrained heads are present before the eval
echo "[$(stamp)] retrained heads on disk:" | tee -a "$LOG"
for h in colorless_duri ulti duri_colored; do
  ls -la "$DIR/${h}_real_baseline.pt" >> "$LOG" 2>&1 && echo "  $h OK" | tee -a "$LOG" \
    || echo "  $h MISSING !!" | tee -a "$LOG"
done

echo "[$(stamp)] clearing stale PIMC-16 eval data + launching PIMC-32 gate" | tee -a "$LOG"
rm -f "$DIR"/mu_*.jsonl "$DIR/TOURNAMENT.md"
: > "$DIR/tournament.log"
PIMC_N=32 bash "$DIR/run_tournament.sh"

echo "===== PIMC-32 full recal DONE $(stamp) — see TOURNAMENT.md =====" | tee -a "$LOG"
