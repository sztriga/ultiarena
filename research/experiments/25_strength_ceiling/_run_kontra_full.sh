#!/bin/zsh
cd /Users/milansimity/Cuccok/kodok/oldtawer
LOG=experiments/25_strength_ceiling/run_kontra_full.log
echo "=== full-ladder kontra (SCORER=kontra, god play, N=3000) START $(date) ===" > $LOG
for k in 0 1; do
  echo ">>> KONTRA=$k (bidder kontra-aware), FLOOR=0.7" >> $LOG
  KONTRA=$k FLOOR=0.7 N=3000 SCORER=kontra DEBIAS_PCTL=0.80 CALIBRATE=1 python3 experiments/24_bidding_loop/harness.py 2>&1 \
    | grep -vE "^ *[0-9]+/3000" | grep -E "METRIC|NON-FLOOR|GP/seat|pass |piros parti|piros ulti " >> $LOG
  echo >> $LOG
done
echo "=== done $(date) ===" >> $LOG
