# Overnight findings — realistic Ulti agent (2026-07-05 → 06)

**Verdict:** the way to beat humans is **not** better bidding nets or more perfect play —
it's **exploiting imperfect opponents in-play**. Tonight I cleaned up the agent so it now
*beats* imperfect defenders on the full 33-contract ladder (nothing dropped), and it's
**cheat-clean**: own hand + public info only (your camera-POV requirement), audited 3× —
bidder, PIMC play, and the new kontra decision. Positive control: a god (perfect-info)
bidder FAILS the same audit, so the test is real. God is a measurement ceiling only.

**The lever map (firmed at N≥500).** Two *tunable* knobs are both only **modest and
comparable**: (1) bidding-net accuracy — perfect info at bid time is worth +7.3 GP/deal but
82% is irreducible (you can't see opponents' hands), so only ~+1.3 is trainable; (2)
play-search depth — more PIMC search helps the opener monotonically (P0 −2.5→−1.6 over
N=4→16) but plateaus, worth ~+1. Cranking either won't beat humans. The lever that matters
is *qualitative*: perfect-INFO play actually **hurts** (the god solver quits on
double-dummy-lost contracts) while uncertain PIMC keeps fighting and **exploits defender
mistakes** — so the headroom is opponent-*modeling*, not a bigger net or deeper search.

**Wins captured.** Confidence FLOOR=0.7 stops declaring unmakeable contracts (god metric
−2.3→−1.4 at N=2000, bleeders removed). Kontra is now wired across the whole ladder, with a
unified realistic+kontra eval (cheat-proof hand-based kontra decisions during PIMC play).
Definitive measurement (N=600): kontra-aware bidding lifts the forced opener **P0 −1.5 →
+0.9 GP/game** (crosses from losing to winning), passing 69% of weak hands that otherwise
get kontra'd double (piros parti −4.7 → the strong remainder +3.8). Under realistic play the
agent as soloist is net-positive (METRIC +1.1, NON-FLOOR +10, piros ulti +30) — it beats
imperfect defenders when it holds a contract.

**Champion #1 (clean):** net bidder + FLOOR=0.7 + kontra-aware + PIMC play, all 33 rungs.
**Next (see FRONTIER.md):** opponent-modeling / exploitative play — the biggest lever and
the real path to beating humans. Bidding-net retraining (~+1.3) is secondary. One residual
bleeder to fix: terített colorless duri (~2% of hands). Detail in NIGHT_LOG.md / results.md.
