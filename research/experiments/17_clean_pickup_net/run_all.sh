#!/bin/bash
# Exp 17 end-to-end orchestrator. Runs sequentially:
#   1. α=0 datagen (1M per contract)
#   2. multi-head training
#   3. tier 1 calibration check
#   4. tier 3 auction (N=3000)
# Logs to /tmp/exp17_run_<step>.log

set -e
cd /Users/milansimity/Cuccok/kodok/oldtawer/experiments/17_clean_pickup_net

echo "=== [1/4] α=0 datagen ===" | tee /tmp/exp17_run.log
date | tee -a /tmp/exp17_run.log
python3 gen_alpha0.py 1000000 2>&1 | tee /tmp/exp17_run_1_datagen.log
echo | tee -a /tmp/exp17_run.log

echo "=== [2/4] training ===" | tee -a /tmp/exp17_run.log
date | tee -a /tmp/exp17_run.log
python3 train.py 1M 2>&1 | tee /tmp/exp17_run_2_train.log
echo | tee -a /tmp/exp17_run.log

echo "=== [3/4] tier 1 calibration ===" | tee -a /tmp/exp17_run.log
date | tee -a /tmp/exp17_run.log
python3 tier1_calibration.py 2>&1 | tee /tmp/exp17_run_3_tier1.log
echo | tee -a /tmp/exp17_run.log

echo "=== [4/4] tier 3 auction (N=3000) ===" | tee -a /tmp/exp17_run.log
date | tee -a /tmp/exp17_run.log
N_DEALS=3000 python3 baseline_tier3.py 2>&1 | tee /tmp/exp17_run_4_tier3.log
echo | tee -a /tmp/exp17_run.log

echo "=== ALL DONE ===" | tee -a /tmp/exp17_run.log
date | tee -a /tmp/exp17_run.log
