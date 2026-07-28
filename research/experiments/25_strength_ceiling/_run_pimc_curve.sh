#!/bin/zsh
# Firm the "play quality is the BIG lever" backbone at N=500 (was noisy N=150).
# Realistic PIMC play (SCORER=pimc), FLOOR=0.7, sweep PIMC_N. Isolates play quality.
cd /Users/milansimity/Cuccok/kodok/oldtawer
LOG=experiments/25_strength_ceiling/run_pimc_curve.log
echo "=== PIMC_N play-lever curve (SCORER=pimc, N=500, FLOOR=0.7) START $(date) ===" > $LOG
for pn in 4 8 16; do
  echo ">>> PIMC_N=$pn" >> $LOG
  FLOOR=0.7 N=500 SCORER=pimc PIMC_N=$pn DEBIAS_PCTL=0.80 CALIBRATE=1 \
    python3 experiments/24_bidding_loop/harness.py 2>&1 \
    | grep -vE "^ *[0-9]+/500" | grep -E "METRIC|NON-FLOOR|GP/seat|pass " >> $LOG
  echo >> $LOG
done
echo "=== done $(date) ===" >> $LOG
