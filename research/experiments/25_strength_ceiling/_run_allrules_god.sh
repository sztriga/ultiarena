#!/bin/zsh
# Stable large-N "all rules" number: full ladder + FLOOR=0.7 + kontra-aware bidding,
# scored with kontra (god play). The champion's god-metric baseline with everything on.
cd /Users/milansimity/Cuccok/kodok/oldtawer
LOG=experiments/25_strength_ceiling/run_allrules_god.log
echo "=== all-rules god (FLOOR=0.7 KONTRA=1 SCORER=kontra N=5000) START $(date) ===" > $LOG
KONTRA=1 FLOOR=0.7 N=5000 SCORER=kontra DEBIAS_PCTL=0.80 CALIBRATE=1 python3 experiments/24_bidding_loop/harness.py 2>&1 \
  | grep -vE "^ *[0-9]+/5000" >> $LOG
echo "=== done $(date) ===" >> $LOG
