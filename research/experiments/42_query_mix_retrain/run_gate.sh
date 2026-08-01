#!/bin/zsh
cd /Users/milansimity/Cuccok/kodok/ultiarena
E=research/experiments/42_query_mix_retrain
MATCHUP=h2h:CAND:FRONTIER N=600 WORKERS=6 python3 $E/tournament.py
MATCHUP=self:CAND N=300 WORKERS=6 python3 $E/tournament.py
MATCHUP=self:FRONTIER N=300 WORKERS=6 python3 $E/tournament.py
echo "GATE ALL DONE"
