#!/bin/bash
# exp38 promotion eval: 3 pairwise h2h + 2 self-play ladders, then the combined report.
set -u
cd /Users/milansimity/Cuccok/kodok/oldtawer
PY=.venv/bin/python
D=experiments/38_promotion_eval
HN=${1:-2000}      # deals per h2h matchup
SN=${2:-3000}      # deals per self-play ladder

for MU in h2h:exp37:FRONTIER h2h:exp36:FRONTIER h2h:exp37:exp36; do
  echo "[promo] $(date +%H:%M:%S) $MU (N=$HN × 3 seatings)…"
  MATCHUP=$MU N=$HN WORKERS=8 $PY $D/tournament.py build > "$D/log_${MU//:/_}.log" 2>&1
done
for MU in self:FRONTIER self:exp37; do
  echo "[promo] $(date +%H:%M:%S) $MU (N=$SN)…"
  MATCHUP=$MU N=$SN WORKERS=8 $PY $D/tournament.py build > "$D/log_${MU//:/_}.log" 2>&1
done

echo "[promo] $(date +%H:%M:%S) building report…"
$PY $D/tournament.py report > "$D/report.log" 2>&1
echo "[promo] $(date +%H:%M:%S) --- RESULTS ---"
cat "$D/RESULTS.md"
echo "[promo] $(date +%H:%M:%S) PROMOTION EVAL DONE."
