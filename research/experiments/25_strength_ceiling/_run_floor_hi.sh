#!/bin/zsh
cd /Users/milansimity/Cuccok/kodok/oldtawer
LOG=experiments/25_strength_ceiling/run_floor_hi.log
echo "=== higher-floor sweep (god, N=3000) START $(date) ===" > $LOG
for floor in 0.6 0.7 0.8 0.9; do
  echo ">>> FLOOR=$floor" >> $LOG
  FLOOR=$floor N=3000 SCORER=god DEBIAS_PCTL=0.80 CALIBRATE=1 python3 experiments/24_bidding_loop/harness.py 2>&1 \
    | grep -vE "^ *[0-9]+/3000" | grep -E "METRIC|NON-FLOOR|GP/seat|pass" >> $LOG
  echo >> $LOG
done
echo "=== done $(date) ===" >> $LOG
