"""Anti-cheating audit. The realistic agent must decide from its OWN info only
(own hand + public) — never opponents' hidden hands, because the target is a
camera-from-POV deployment where those cards are simply not observable.

Behavioural test (no code-tracing needed): fix the soloist's 12 cards, VARY the
opponents' hidden hands; a legal bidder's decision MUST NOT change. The god-
provider is the positive control — it SHOULD change (it cheats), which proves the
test is actually sensitive. god is ceiling-only; it never ships in the agent.

Env: DEALS.
"""
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/23_bidding_integration",
          "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/14_minigame_bid_eval",
          "/Users/milansimity/Cuccok/kodok/oldtawer"):
    if p not in sys.path:
        sys.path.insert(0, p)

from auction import net_bid_fn                # noqa: E402
from provider import NetProvider              # noqa: E402
from god_provider import make_god_bid_fn      # noqa: E402
from _lib import deal_12_10_10, fresh_deck    # noqa: E402

DEALS = int(os.environ.get("DEALS", "25"))


def opp_variations(cards12, n, seed):
    held = {c.id for c in cards12}
    pool = [c for c in fresh_deck() if c.id not in held]      # the other 20 cards
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = pool[:]
        rng.shuffle(p)
        out.append((p[:10], p[10:20]))
    return out


def _key(r):
    if r is None:
        return "PASS"
    return (r[1].name, r[2], tuple(sorted(c.id for c in r[3])))


def audit(make_bid_fn, label):
    bf = make_bid_fn()
    leaks = 0
    for s in range(DEALS):
        sol12, _d1, _d2 = deal_12_10_10(1_000 + s)
        cards = list(sol12)
        decisions = {_key(bf(cards, None, others))
                     for others in opp_variations(cards, 5, s)}
        if len(decisions) > 1:
            leaks += 1
    verdict = "LEAK — decision depends on hidden hands (CHEATING)" if leaks else \
              "clean — decision invariant to hidden hands"
    print(f"{label:<26} {leaks:>3}/{DEALS} deals changed  → {verdict}", flush=True)
    return leaks


if __name__ == "__main__":
    print(f"=== anti-cheating audit | DEALS={DEALS} ===", flush=True)
    net_leaks = audit(lambda: net_bid_fn(NetProvider(calibrate=True)),
                      "NET bidder (the agent)")
    god_leaks = audit(make_god_bid_fn, "GOD bidder (control)")
    print()
    ok = (net_leaks == 0 and god_leaks > 0)
    print("AUDIT PASS — agent clean, test sensitive" if ok else
          f"AUDIT ISSUE — net_leaks={net_leaks} (want 0), god_leaks={god_leaks} (want >0)")
    sys.exit(0 if ok else 1)
