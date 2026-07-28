#!/bin/bash
cd /Users/milansimity/Cuccok/kodok/oldtawer/experiments/36_betli_defense
PY=/Users/milansimity/Cuccok/kodok/oldtawer/.venv/bin/python
export WORKERS=8 EPOCHS=60 PIMC_N=16
mkdir -p models; LOG=sweep.log
echo "=== BETLI POLICY SWEEP $(date) ===" > $LOG
run() {
  echo "[$(date +%H:%M:%S)] TRAIN $1 (arch=$2 h=$3 wd=$4 drop=$5)" >> $LOG
  ARCH=$2 HIDDEN=$3 WD=$4 DROPOUT=$5 MODEL_OUT=models/$1.pt $PY policy.py train >> $LOG 2>&1
  MODEL_IN=models/$1.pt N=3500 $PY policy.py eval >> $LOG 2>&1
  echo "  >>> RESULT $1: $(grep -h 'NET  defense\|PIMC defense\|Δ =' EVAL.md | tr '\n' ' ')" >> $LOG
}
run mlp256   mlp  256 1e-4 0.2
run mlp512   mlp  512 2e-4 0.3
run mlp384   mlp  384 3e-4 0.3
run wide256  wide 256 1e-4 0.25
run deep256  deep 256 1e-4 0.25
echo "=== SWEEP DONE $(date) ===" >> $LOG
