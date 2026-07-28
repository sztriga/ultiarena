#!/bin/bash
set -u
cd /Users/milansimity/Cuccok/kodok/oldtawer
PY=.venv/bin/python; D=experiments/38_promotion_eval
for MU in self:FRONTIER self:exp37; do
  echo "[ladder] $(date +%H:%M:%S) $MU (N=3000, resolved contracts)…"
  MATCHUP=$MU N=3000 WORKERS=8 $PY $D/tournament.py build > "$D/log_${MU//:/_}.log" 2>&1
done
echo "[ladder] $(date +%H:%M:%S) rebuilding report…"
$PY $D/tournament.py report > "$D/report.log" 2>&1
echo "[ladder] $(date +%H:%M:%S) --- LADDER (fixed) ---"; cat "$D/RESULTS.md"
echo "[ladder] $(date +%H:%M:%S) DONE."
