#!/bin/bash
# exp40 — full-slate realistic recalibration: datagen + train for every base-event head.
# Sequential, cheapest-first (so an early status ping shows completed heads), duri_colored
# (the expensive leak head) near the end. Each head: datagen -> its .npz, then train -> its
# _real_baseline.pt + _real_isotonic.npz + a reliability/god-comparison report. Robust: a
# failed head is logged and skipped, never aborts the run. All output tees to per-head logs
# + a master heartbeat with timestamps.
set -u
cd /Users/milansimity/Cuccok/kodok/oldtawer
PY=.venv/bin/python
DIR=experiments/40_realistic_recal
LOG=$DIR/run.log
export WORKERS=8
export PIMC_N=8            # datagen defense strength (deployed play uses 16; the GP tournament
                          # validates the head vs the real 16-sample engine — see RESULTS.md)

# head:N:SEED_BASE  (cheapest first; duri_colored last-but-one; ulti last)
JOBS=(
  "colorless_duri:40000:400000000"
  "reach100_40:25000:450000000"
  "reach100_20:15000:500000000"
  "parti:15000:550000000"
  "duri_colored:10000:600000000"
  "ulti:20000:650000000"
)

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
echo "===== exp40 run START $(stamp)  workers=$WORKERS pimc_n=$PIMC_N =====" | tee -a "$LOG"

for job in "${JOBS[@]}"; do
  IFS=":" read -r HEAD N SEED <<< "$job"
  echo "[$(stamp)] >>> $HEAD  datagen N=$N seed=$SEED" | tee -a "$LOG"
  if HEAD=$HEAD N=$N SEED_BASE=$SEED $PY $DIR/datagen.py > "$DIR/${HEAD}_datagen.log" 2>&1; then
    tail -2 "$DIR/${HEAD}_datagen.log" | sed "s/^/    /" | tee -a "$LOG"
  else
    echo "[$(stamp)] !!! $HEAD datagen FAILED (see ${HEAD}_datagen.log) — skipping" | tee -a "$LOG"
    continue
  fi
  echo "[$(stamp)] >>> $HEAD  train" | tee -a "$LOG"
  if HEAD=$HEAD $PY $DIR/train.py > "$DIR/${HEAD}_train.log" 2>&1; then
    grep -E "REALISTIC head|GOD head|realistic head mean" "$DIR/${HEAD}_train.log" | sed "s/^/    /" | tee -a "$LOG"
  else
    echo "[$(stamp)] !!! $HEAD train FAILED (see ${HEAD}_train.log)" | tee -a "$LOG"
  fi
  echo "[$(stamp)] <<< $HEAD done" | tee -a "$LOG"
done

echo "===== exp40 run DONE $(stamp) =====" | tee -a "$LOG"
