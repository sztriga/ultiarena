#!/bin/bash
set -u
cd /Users/milansimity/Cuccok/kodok/oldtawer
PY=.venv/bin/python; D=experiments/39_rebetli
echo "[exp39] $(date +%H:%M:%S) self:FRONTIER (N=6000 — baseline ladder + negative-games)…"
MATCHUP=self:FRONTIER N=6000 WORKERS=8 $PY $D/tournament.py build > "$D/log_self_FRONTIER.log" 2>&1
echo "[exp39] $(date +%H:%M:%S) self:REBETLI (N=6000 — rebetli freq/GP/displacement)…"
MATCHUP=self:REBETLI N=6000 WORKERS=8 $PY $D/tournament.py build > "$D/log_self_REBETLI.log" 2>&1
echo "[exp39] $(date +%H:%M:%S) h2h:REBETLI:FRONTIER (N=3000 — game-points impact)…"
MATCHUP=h2h:REBETLI:FRONTIER N=3000 WORKERS=8 $PY $D/tournament.py build > "$D/log_h2h.log" 2>&1
echo "[exp39] $(date +%H:%M:%S) report…"
$PY $D/tournament.py report > "$D/report.log" 2>&1
echo "[exp39] $(date +%H:%M:%S) --- RESULTS ---"; cat "$D/RESULTS.md"
echo "[exp39] $(date +%H:%M:%S) DONE."
