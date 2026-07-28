#!/bin/zsh
# Firm the perception-headroom SPLIT at N=600 (was N=150, wide CI). Reproduces Phase 1b
# with the standard net config, just at 4x the deals: total (net->god), TRAINABLE
# (net->PIMC marginal), irreducible (PIMC->god info gap).
cd /Users/milansimity/Cuccok/kodok/oldtawer
LOG=experiments/25_strength_ceiling/run_ceiling_firm.log
echo "=== perception-split CONFIRM (N=600, N_DET=8) START $(date) ===" > $LOG
N=600 N_DET=8 DEBIAS_PCTL=0.80 CALIBRATE=1 \
  python3 experiments/25_strength_ceiling/ceiling_split.py >> $LOG 2>&1
echo "=== done $(date) ===" >> $LOG
