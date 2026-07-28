#!/bin/zsh
cd /Users/milansimity/Cuccok/kodok/oldtawer
LOG=experiments/25_strength_ceiling/run_champion.log
echo "=== champion realistic (net + FLOOR=0.7 + PIMC play, N=400) START $(date) ===" > $LOG
FLOOR=0.7 N=400 SCORER=pimc PIMC_N=16 DEBIAS_PCTL=0.80 CALIBRATE=1 python3 experiments/24_bidding_loop/harness.py 2>&1 \
  | grep -vE "^ *[0-9]+/400" >> $LOG
echo "=== done $(date) ===" >> $LOG
