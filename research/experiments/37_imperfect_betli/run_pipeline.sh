#!/bin/bash
# exp37 orchestration: wait for datagen → train head → main tournament (PIMC) → robustness (god).
# Each stage logs to its own file; heartbeats so milan can peek mid-run.
set -u
cd /Users/milansimity/Cuccok/kodok/oldtawer
PY=.venv/bin/python
D=experiments/37_imperfect_betli
DGPID=${1:-34573}

echo "[pipe] $(date +%H:%M:%S) waiting for datagen (pid $DGPID)…"
while kill -0 "$DGPID" 2>/dev/null; do sleep 30; done
if [ ! -f "$D/betli_real.npz" ]; then echo "[pipe] ERROR: betli_real.npz missing after datagen"; exit 1; fi
echo "[pipe] $(date +%H:%M:%S) datagen done. training head…"

EPOCHS=30 $PY $D/train.py > $D/train.log 2>&1
if [ ! -f "$D/betli_real_baseline.pt" ]; then echo "[pipe] ERROR: head not saved"; exit 1; fi
echo "[pipe] $(date +%H:%M:%S) head trained. --- train report tail ---"
tail -20 $D/train.log

echo "[pipe] $(date +%H:%M:%S) MAIN tournament (DEF=pimc, deployed table)…"
N=1500 WORKERS=8 DEF=pimc OUT=$D/tournament_pimc.jsonl $PY $D/tournament.py build > $D/tour_pimc.log 2>&1
DEF=pimc OUT=$D/tournament_pimc.jsonl $PY $D/tournament.py analyze >> $D/tour_pimc.log 2>&1
echo "[pipe] $(date +%H:%M:%S) --- MAIN result ---"; cat $D/TOURNAMENT_pimc.md

echo "[pipe] $(date +%H:%M:%S) ROBUSTNESS tournament (DEF=god, perfect defenders punish bluffs)…"
N=1500 WORKERS=8 DEF=god OUT=$D/tournament_god.jsonl $PY $D/tournament.py build > $D/tour_god.log 2>&1
DEF=god OUT=$D/tournament_god.jsonl $PY $D/tournament.py analyze >> $D/tour_god.log 2>&1
echo "[pipe] $(date +%H:%M:%S) --- ROBUSTNESS result ---"; cat $D/TOURNAMENT_god.md

echo "[pipe] $(date +%H:%M:%S) PIPELINE DONE."
