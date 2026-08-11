"""Regenerate the web UI golden fixtures (apps/web/src/test/fixtures/).

The UI golden test replays a RECORDED game through a mocked api in strict step
order, driving the component the exact way the recording was made: open, bid the
lowest rung, pass twice, kontra everything offered, play the first legal card ten
times, open the analysis. This script produces such a recording from the live
backend — used at the 2026-08-11 suit re-encode (the old fixture's ids and deal
became invalid) and any future protocol change.

It SEARCHES seeds for a game with the exact interaction shape the test drives:
  /play/new → /play/bid → /play/pass ×2 → (10× /play/move + ≥1 /play/kontra) → /play/analysis

Run:  python tests/golden/capture_web_fixture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

OUT = _REPO / "apps" / "web" / "src" / "test" / "fixtures"


def _try_seed(seed: int):
    from apps.api import play as P
    from apps.api.engine import _sessions, _sessions_lock

    steps = []

    def rec(action, endpoint, request, response):
        steps.append({"action": action, "endpoint": endpoint,
                      "request": request, "response": response})

    snap = P.play_new(P.NewRequest(seat=0, seed=seed))
    gid = snap["game_id"]
    rec("new", "/play/new", {"seat": 0}, snap)
    try:
        auc = snap["auction"]
        real = [b for b in (auc["legal_bids"] or []) if b["rung_index"] >= 0]
        if not (auc["opening"] and real):
            return None
        b = real[0]
        req = {"game_id": gid, "rung_index": b["rung_index"], "bid_index": b["bid_index"],
               "trump": b["trump_options"][0] if len(b["trump_options"]) == 1 else None,
               "discard_ids": snap["talon_ids"][:2]}
        snap = P.play_bid(P.BidRequest(**req))
        rec("bid", "/play/bid", req, snap)

        # pass (Passz / Elfogadom — same button) until play starts; the UI test
        # drives these fixture-led, so any count works
        n_pass = 0
        while snap["phase"] == "bid" and n_pass < 5:
            req = {"game_id": gid, "discard_ids": []}
            snap = P.play_pass(P.PassRequest(**req))
            rec("pass", "/play/pass", req, snap)
            n_pass += 1
        if snap["phase"] not in ("play", "kontra") or n_pass == 0:
            return None

        moves = kontras = 0
        for _ in range(60):
            if snap["phase"] == "kontra":
                req = {"game_id": gid, "action": "double"}
                snap = P.play_kontra(P.KontraRequest(game_id=gid, action="double"))
                rec("kontra_double", "/play/kontra", req, snap)
                kontras += 1
            elif snap["phase"] == "play" and snap.get("legal_card_ids"):
                cid = snap["legal_card_ids"][0]
                req = {"game_id": gid, "card_id": cid}
                snap = P.play_move(P.MoveRequest(**req))
                rec("move", "/play/move", req, snap)
                moves += 1
            elif snap["phase"] == "done":
                break
            else:
                return None
        if snap["phase"] != "done" or moves != 10 or kontras < 1:
            return None

        req = {"game_id": gid}
        resp = P.play_analysis(P.AnalysisRequest(**req))
        rec("analysis", "/play/analysis", req, resp)
        return steps
    finally:
        with _sessions_lock:
            _sessions.pop(gid, None)


def main():
    for seed in range(11, 5000):
        steps = _try_seed(seed)
        if steps is None:
            continue
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump({"seed": seed, "steps": steps}, open(OUT / "game_s0_seed14.json", "w"),
                  separators=(",", ":"))
        n_moves = sum(1 for s in steps if s["action"] == "move")
        print(f"fixture recorded: seed={seed}, {len(steps)} steps, {n_moves} moves, "
              f"contract={steps[-1]['response'].get('contract')}")
        break
    else:
        raise SystemExit("no seed matched the fixture shape in range")

    from apps.api import puzzle as PZ
    snap = PZ.puzzle_new()
    json.dump(snap, open(OUT / "puzzle_new.json", "w"), separators=(",", ":"))
    PZ._sessions.clear()
    print("puzzle fixture recorded")


if __name__ == "__main__":
    main()
