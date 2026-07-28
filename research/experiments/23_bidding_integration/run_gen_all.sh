#!/bin/zsh
# Generate all 3 new base-event datasets sequentially (background job).
# Balanced alphas (biased for training; deployment calibration flagged in PLAN).
cd /Users/milansimity/Cuccok/kodok/oldtawer
LOG=experiments/23_bidding_integration/run_gen.log
echo "=== base-event datagen START $(date) ===" > $LOG
caffeinate -i -s -m zsh -c '
  cd /Users/milansimity/Cuccok/kodok/oldtawer
  HEAD=reach100_40  ALPHA=1.0 N=1000000 WORKERS=8 python3 experiments/23_bidding_integration/gen_base_events.py
  HEAD=duri_colored ALPHA=4.0 N=1000000 WORKERS=8 python3 experiments/23_bidding_integration/gen_base_events.py
  HEAD=reach100_20  ALPHA=3.0 N=1000000 WORKERS=8 python3 experiments/23_bidding_integration/gen_base_events.py
' >> $LOG 2>&1
echo "=== base-event datagen DONE $(date) ===" >> $LOG
