"""Golden-transcript harness for the Ulti play engine (behavior-preservation net).

Drives full deterministic games (scripted human: open lowest rung / play first legal
card / seed-parity kontra) via the real apps.api.play endpoint functions, and records
the COMPLETE transcript per (seat, seed): the auction bid sequence, every card played,
kontra decisions, and the final score. The AI is fully seeded (deal_12_10_10(seed),
random.Random(seed+salt), deterministic torch inference), so with pinned env the output
is reproducible byte-for-byte.

Run against any repo that has apps/api/play.py:
    REPO_ROOT=/path/to/repo python golden_capture.py > transcripts.json
Compare two repos:
    diff <(REPO_ROOT=oldtawer python golden_capture.py) <(REPO_ROOT=trickster python golden_capture.py)
"""
import hashlib
import json
import os
import sys

# ── pin every env knob that affects play BEFORE importing play.py (it reads at import) ──
_PINNED = {
    "FLOOR": "0.80", "DEBIAS_PCTL": "0.85", "DURI_TERIT_MULT": "0.3", "KONTRA": "1",
    "REBETLI_FLOOR": "0.90",
    "EXPLOIT": "1", "EXPLOIT_EPS": "0.15", "EXPLOIT_NW": "16", "EXPLOIT_FRAC": "0.10",
    "BETLI_REAL_BID": "1", "BETLI_DEF": "1", "REBETLI_REAL_BID": "1",
    "PLAY_PIMC_N": "16", "PLAY_KONTRA_NDET": "6",
}
for k, v in _PINNED.items():
    os.environ[k] = v

_REPO = os.environ.get("REPO_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from apps.api import play as P                                                # noqa: E402
from apps.api.play import (NewRequest, BidRequest, PassRequest,               # noqa: E402
                           MoveRequest, KontraRequest)

# fixed, reproducible coverage matrix (seat rotation × a seed sweep that surfaces
# parti / ulti / 40-100 / duri / betli / kontra)
MATRIX = [(s, sd) for s in (0, 1, 2) for sd in range(11, 19)]


def _human_bid_or_pass(snap, gid):
    auc = snap["auction"]
    if not auc["opening"]:
        return P.play_pass(PassRequest(game_id=gid))
    # lowest REAL contract (skip the passz sentinel at rung_index -1)
    real = [b for b in (auc["legal_bids"] or []) if b["rung_index"] >= 0]
    if not real:
        return P.play_pass(PassRequest(game_id=gid))
    lb = real[0]
    trump = lb["trump_options"][0] if lb["trump_options"] else None
    return P.play_bid(BidRequest(game_id=gid, rung_index=lb["rung_index"],
                                 trump=trump, discard_ids=snap["talon_ids"][:2]))


def _capture(seat, seed, kontra_action):
    """Return a fully-determined transcript dict for one game (or a pass/deadend)."""
    snap = P.play_new(NewRequest(seat=seat, seed=seed))
    gid = snap["game_id"]
    bids = []
    guard = 0
    while snap["phase"] == "bid":
        guard += 1
        if guard > 25:
            raise SystemExit("bid loop stuck")
        snap = _human_bid_or_pass(snap, gid)
    # auction bid sequence (public history: pid, contract, trump, rung)
    for h in snap.get("auction", {}).get("history", []) if isinstance(snap.get("auction"), dict) else []:
        bids.append(h)
    if snap["phase"] == "passed":
        return {"seat": seat, "seed": seed, "outcome": "PASSED"}

    kontra_log = []
    guard = 0
    while snap["phase"] in ("play", "kontra"):
        guard += 1
        if guard > 50:
            raise SystemExit("play loop stuck")
        if snap["phase"] == "kontra":
            kontra_log.append({"ply": len(snap["history"]),
                               "role": snap["kontra"]["pending"]["role"],
                               "action": kontra_action})
            snap = P.play_kontra(KontraRequest(game_id=gid, action=kontra_action))
        else:
            snap = P.play_move(MoveRequest(game_id=gid, card_id=snap["legal_card_ids"][0]))

    r = snap["result"]
    # the COMPLETE play sequence — every ply's player + card id (the AI's exact plays)
    plays = [(h["player_id"], h["card"]["id"]) for h in snap["history"]]
    return {
        "seat": seat, "seed": seed, "outcome": "played",
        "contract": snap["contract"], "trump": snap["trump"],
        "human_play_index": snap["human_play_index"],
        "bids": [(b.get("pid"), b.get("contract"), b.get("rung_index")) for b in bids],
        "kontra": kontra_log, "kontra_level": r["kontra_level"],
        "plays": plays,
        "result": {k: r[k] for k in ("winner", "made", "sol_gp_per_def", "human_gp",
                                     "user_won", "soloist_seat", "seat_gp")},
    }


def main():
    cases = []
    for seat, seed in MATRIX:
        ka = "double" if (seed % 2 == 0) else "pass"
        cases.append(_capture(seat, seed, ka))
    blob = json.dumps(cases, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(blob.encode()).hexdigest()
    played = sum(1 for c in cases if c["outcome"] == "played")
    sys.stderr.write(f"golden: {len(cases)} cases, {played} played, sha256={digest[:16]}\n")
    print(json.dumps({"sha256": digest, "n": len(cases), "cases": cases},
                     sort_keys=True, ensure_ascii=False, indent=None))


if __name__ == "__main__":
    main()
