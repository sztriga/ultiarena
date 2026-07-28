#!/bin/bash
# exp37 overnight betli-bleed investigation. Waits for the v2 tournament (avoid CPU contention),
# then runs the head-calibration probe + faithful self-play bleed + analysis.
set -u
cd /Users/milansimity/Cuccok/kodok/oldtawer
PY=.venv/bin/python
D=experiments/37_imperfect_betli
V2PID=${1:-35450}

echo "[bleed] $(date +%H:%M:%S) waiting for v2 tournament (pid $V2PID)…"
while kill -0 "$V2PID" 2>/dev/null; do sleep 60; done
echo "[bleed] $(date +%H:%M:%S) v2 done. --- v2 tournament tail ---"
tail -14 $D/rerun_v2_pipeline.log

echo "[bleed] $(date +%H:%M:%S) PART A: head calibration vs god (N=25000)…"
N=25000 WORKERS=8 $PY $D/betli_bleed.py calib > $D/bleed_calib.log 2>&1
echo "[bleed] $(date +%H:%M:%S) PART B: faithful self-play bleed (N=25000)…"
N=25000 WORKERS=8 $PY $D/betli_bleed.py bleed > $D/bleed_selfplay.log 2>&1
echo "[bleed] $(date +%H:%M:%S) analyzing…"
$PY $D/betli_bleed.py analyze > $D/bleed_analyze.log 2>&1
echo "[bleed] $(date +%H:%M:%S) --- BLEED REPORT ---"
cat $D/BLEED.md
echo "[bleed] $(date +%H:%M:%S) OVERNIGHT DONE."
