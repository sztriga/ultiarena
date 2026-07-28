#!/bin/zsh
# N=600 confirmation of the DEFINITIVE realistic+kontra champion — firms up the noisy
# N=200 headline (P0 +0.40 -> +3.34). Same config: FLOOR=0.7 + PIMC play + hand-based kontra.
cd /Users/milansimity/Cuccok/kodok/oldtawer
LOG=experiments/25_strength_ceiling/run_kpimc_confirm.log
echo "=== realistic+kontra champion CONFIRM (SCORER=kpimc, N=600) START $(date) ===" > $LOG
for k in 0 1; do
  echo ">>> KONTRA=$k (bidder), FLOOR=0.7, PIMC play + hand-based kontra" >> $LOG
  KONTRA=$k FLOOR=0.7 N=600 SCORER=kpimc PIMC_N=16 KONTRA_NDET=6 DEBIAS_PCTL=0.80 CALIBRATE=1 \
    python3 experiments/24_bidding_loop/harness.py 2>&1 \
    | grep -vE "^ *[0-9]+/600" | grep -E "METRIC|NON-FLOOR|GP/seat|pass |piros parti|piros ulti " >> $LOG
  echo >> $LOG
done
echo "=== done $(date) ===" >> $LOG
