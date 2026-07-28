# Ulti teaching — Never give up a "lost" hand (fight for the defender's mistake)

*Part of the teaching corpus (see also experiments/27_kontra_revamp/TEACHING.md for the
make-probability tables, and reference_ulti_makeprobs).*

## The lesson
A hand that is theoretically lost — one that *perfect* defense would always beat — is **not lost
in practice**, because real defenders make mistakes. The right play in a lost position is NOT to
resign, but to play the line that gives the defender the **most chances to go wrong**.

## The numbers (measured, betli, N=1000 deals)
We took betli hands that a *perfect* (double-dummy) defense would always beat, and played them
against a strong-but-fallible defender (plays well, but errs ~30% of the time — a rough human proxy):

| soloist's approach | betli win-rate (theoretically-lost hands) |
|---|---|
| "resign" (double-dummy optimal — treats all lost lines as equal) | ~2% |
| standard careful play (PIMC) | ~2% |
| **fight for the mistake (play the trickiest line)** | **~5%** |

So **~1 in 20 "lost" betlis is still won** just by choosing the line that's hardest to defend —
2–3× what you get by playing "correctly-but-resigned." Against weaker opponents the gap is bigger
(it grows with how often the defender errs).

## Why this works (and why a pure solver misses it)
A perfect-information solver assumes the defender also plays perfectly, so on a lost hand *every*
move loses — they all look equally hopeless, and it picks one at random. It cannot tell "loses
cleanly" from "loses unless the defender finds the one hard defense." A strong human — and our
exploit engine — instead asks: *which line forces the defender to find the most, and hardest,
correct plays?* Set the trap; often they miss it.

## Practical rules of thumb for the student
- **In a lost contract, complicate.** Lead the suit/keep the holding that forces the defender into
  the most decisions. Don't cash out into a clean, easy defense.
- **The mirror image (defending):** when the soloist is in trouble and starts making things messy,
  slow down — that's exactly when they're hoping you'll slip. Count carefully.
- **This is where matches are won at the club level:** experts and the engine agree the *bidding*
  and *makeable* hands are mostly routine; the edge is in (a) defending well (not gifting the
  trap) and (b) squeezing the last few % out of bad hands.

## Companion insight — the perfect-vs-real gap (from the kontra tables)
The same idea in reverse: betli is made **66% under perfect defense but 86% in real play** — that
20-point gap is entirely defenders letting winnable-only-with-precision hands through. Betli is the
Ulti contract that most rewards *defensive skill* (see TEACHING.md §3). Trumps decide the trick
contracts (ulti: a 4-trump defender beats it ~2/3 of the time); betli decides on pure card-play.
