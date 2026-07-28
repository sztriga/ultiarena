#!/bin/zsh
cd /Users/milansimity/Cuccok/kodok/oldtawer
LOG=experiments/25_strength_ceiling/run_pimc_floor.log
echo "=== PIMC floor confirm START $(date) ===" > $LOG
caffeinate -i -s zsh -c '
  cd /Users/milansimity/Cuccok/kodok/oldtawer
  for floor in 0 0.7; do
    echo ">>> FLOOR=$floor SCORER=pimc"
    FLOOR=$floor N=150 SCORER=pimc PIMC_N=16 DEBIAS_PCTL=0.80 CALIBRATE=1 python3 experiments/24_bidding_loop/harness.py 2>&1 \
      | grep -vE "^ *[0-9]+/150" | grep -E "METRIC|NON-FLOOR|GP/seat| 20-100 |rebetli| betli |teritett rebetli"
    echo
  done
' >> $LOG 2>&1
echo "=== done $(date) ===" >> $LOG
