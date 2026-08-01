#!/bin/zsh
cd /Users/milansimity/Cuccok/kodok/ultiarena
HEAD=duri_colored N_QUERY=120000 N_BIAS=40000 N_CALIB=25000 WORKERS=8 \
  python3 research/experiments/42_query_mix_retrain/gen_mix.py
HEAD=reach100_20 N_QUERY=60000 N_BIAS=30000 N_CALIB=15000 WORKERS=8 \
  python3 research/experiments/42_query_mix_retrain/gen_mix.py
echo "DATAGEN DONE"
