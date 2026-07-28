#!/bin/zsh
# DEFINITIVE realistic all-rules champion: FLOOR=0.7 + PIMC play + hand-based kontra
# decision (cheat-proof). Compare KONTRA=0 (not kontra-aware bidding) vs KONTRA=1
# (passes weak hands) — under the realistic+kontra scorer.
cd /Users/milansimity/Cuccok/kodok/oldtawer
LOG=experiments/25_strength_ceiling/run_kpimc_champion.log
echo "=== realistic+kontra champion (SCORER=kpimc, N=200) START $(date) ===" > $LOG
for k in 0 1; do
  echo ">>> KONTRA=$k (bidder), FLOOR=0.7, PIMC play + hand-based kontra" >> $LOG
  KONTRA=$k FLOOR=0.7 N=200 SCORER=kpimc PIMC_N=16 KONTRA_NDET=6 DEBIAS_PCTL=0.80 CALIBRATE=1 \
    python3 experiments/24_bidding_loop/harness.py 2>&1 \
    | grep -vE "^ *[0-9]+/200" | grep -E "METRIC|NON-FLOOR|GP/seat|pass |piros parti|piros ulti " >> $LOG
  echo >> $LOG
done
echo "=== done $(date) ===" >> $LOG
