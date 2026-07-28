#!/bin/bash
# exp37 v2 re-run: realistic prob now drives PLAIN betli only (rebetli/terit keep god prob).
# Head unchanged; only the bidder integration changed → re-run both tournaments.
set -u
cd /Users/milansimity/Cuccok/kodok/oldtawer
PY=.venv/bin/python
D=experiments/37_imperfect_betli

echo "[v2] $(date +%H:%M:%S) MAIN tournament (DEF=pimc)…"
N=1500 WORKERS=8 DEF=pimc OUT=$D/tournament_pimc_v2.jsonl $PY $D/tournament.py build > $D/tour_pimc_v2.log 2>&1
DEF=pimc OUT=$D/tournament_pimc_v2.jsonl $PY $D/tournament.py analyze >> $D/tour_pimc_v2.log 2>&1
echo "[v2] $(date +%H:%M:%S) --- MAIN (pimc) v2 ---"; sed -n '/## Headline/,/what C bid/p' $D/tour_pimc_v2.log | tail -8

echo "[v2] $(date +%H:%M:%S) ROBUSTNESS tournament (DEF=god)…"
N=1500 WORKERS=8 DEF=god OUT=$D/tournament_god_v2.jsonl $PY $D/tournament.py build > $D/tour_god_v2.log 2>&1
DEF=god OUT=$D/tournament_god_v2.jsonl $PY $D/tournament.py analyze >> $D/tour_god_v2.log 2>&1
echo "[v2] $(date +%H:%M:%S) --- ROBUSTNESS (god) v2 ---"; sed -n '/## Headline/,/what C bid/p' $D/tour_god_v2.log | tail -8
echo "[v2] $(date +%H:%M:%S) V2 DONE."
