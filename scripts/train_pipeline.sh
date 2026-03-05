#!/usr/bin/env bash
#
# Training pipeline: rook → morpheus + tournament.
#
# Rook (512×6) trains from scratch, morpheus warm-starts from rook.
#
# Usage:
#   ./scripts/train_pipeline.sh              # default 6 workers
#   ./scripts/train_pipeline.sh 4            # custom worker count
#   GAMES=300 ./scripts/train_pipeline.sh    # custom tournament games

set -euo pipefail

WORKERS="${1:-6}"
TOURNEY_GAMES="${GAMES:-500}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ULTI TRAINING PIPELINE                                     ║"
echo "║  rook → morpheus + tournament                               ║"
echo "║  Workers: $WORKERS                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Rook: 512×6 net, from scratch ──
echo "━━━ Stage 1/2: rook (from scratch) ━━━"
python3 scripts/train_e2e.py rook \
    --workers "$WORKERS"

# ── 2. Morpheus: warm-start from rook ──
echo "━━━ Stage 2/2: morpheus (from rook) ━━━"
python3 scripts/train_e2e.py morpheus \
    --from rook \
    --workers "$WORKERS"

# ── Tournament ──
echo ""
echo "━━━ TOURNAMENT ━━━"
python3 scripts/tournament.py trinity rook morpheus random \
    --games "$TOURNEY_GAMES" \
    --workers "$WORKERS"

echo ""
echo "Pipeline complete."
