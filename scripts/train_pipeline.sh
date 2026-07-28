#!/usr/bin/env bash
#
# Training pipeline: bishop → trinity + tournament.
#
# Bishop (384×4) trains from scratch, trinity warm-starts from bishop.
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
echo "║  bishop → trinity + tournament                              ║"
echo "║  Workers: $WORKERS                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Bishop: 384×4 net, from scratch ──
echo "━━━ Stage 1/2: bishop (from scratch) ━━━"
python3 scripts/train_e2e.py bishop \
    --workers "$WORKERS"

# ── 2. Trinity: warm-start from bishop ──
echo "━━━ Stage 2/2: trinity (from bishop) ━━━"
python3 scripts/train_e2e.py trinity \
    --from bishop \
    --workers "$WORKERS"

# ── Tournament ──
echo ""
echo "━━━ TOURNAMENT ━━━"
python3 scripts/tournament.py bishop trinity random \
    --games "$TOURNEY_GAMES" \
    --workers "$WORKERS"

echo ""
echo "Pipeline complete."
