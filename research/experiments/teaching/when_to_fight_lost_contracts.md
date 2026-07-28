# Ulti teaching — Which "lost" contracts are worth fighting for?

*Companion to `fighting_lost_hands.md` (never give up a lost hand) and
`27_kontra_revamp/TEACHING.md` (the make-probability tables). This one is about **choosing your
battles**: a lost hand is worth fighting only when a defender's mistake can actually rescue it.*

## The one idea

`fighting_lost_hands.md` taught: don't resign a theoretically-lost hand — play the trickiest line
and hope the defender slips. True. But there's a second half to the lesson that separates a strong
player from a stubborn one:

> **How much a mistake helps you depends on the contract.** In some contracts a *single* defender
> slip hands you the whole game. In others, one slip changes nothing — you needed a specific card
> to fall, and it didn't. Fight hard in the first kind; in the second, don't burn energy (or give
> away information) chasing a miracle that a random error can't deliver.

Think of it as **"trap headroom"**: how much a fallible opponent can gift you, above what perfect
defense would allow.

## High trap headroom — **Betli** (fight hard)

Betli = take **zero** tricks. To *win*, the soloist needs the defenders to never once force a
trick on them. The soloist can lose in a hundred small ways — which means the defender must be
**precise on every trick**. One loose discard, one wrong lead, and the soloist is through.

- Perfect defense beats a beatable betli ~100% of the time; **real** defense lets ~1 in 20 of those
  through just by the soloist playing the trickiest line (measured — see `fighting_lost_hands.md`).
- There are no trumps, so there's nothing to "count" — it's pure card-reading, and humans slip.

**Rule for the soloist:** a lost betli is *always* worth fighting. Keep the awkward holdings, make
the defenders guess, never cash into a clean line.

## Low trap headroom — **Ulti / Durchmars** (usually don't)

A trick contract is **specific**: an ulti needs the trump 7 to win the *last* trick; a durchmars
needs *every* trick. If the cards lie so that perfect defense stops you, a *random* defender
mistake elsewhere usually **doesn't create the one trick you're missing**. You lose by the same
margin whether you fought or folded — and a bukott (failed) ulti or duri loses a *lot* (the stake
is big, and kontra multiplies it 2–3–5×).

- A lost ulti is lost because a defender holds a trump that beats your 7. Them misplaying a side
  suit doesn't change that. The headroom is small.
- Worse, thrashing around on a hopeless trick contract can *hand information* to good defenders and
  cost you overtricks-worth of points on the parti side.

**Rule for the soloist:** if the key trick is genuinely gone, play it straight and minimize the
bleed. Save the trickery for the rare line where a specific defender error *can* still give you the
trick you need (e.g. an unguarded trump, a suit they might mis-block).

## Middle ground — **Párti and the 100s**

Párti (card points) and the 40-100 / 20-100 games are *graded*: you don't just win or lose, you win
or lose **by an amount**. Here fighting always has *some* value — every extra card-point a mistake
gives you is real GP, even on a hand you'll lose overall. So "fight for the mistake" applies, but
the payoff is incremental (a few points), not all-or-nothing like betli.

## The student's decision table

| Contract | If it's lost under perfect defense… | Why |
|---|---|---|
| **Betli** | **Fight hard** — trickiest line every trick | any single slip wins it for you |
| **Durchmars** | Fight only for a *specific* missing trick | you need *all* tricks; random slips rarely give the one you lack |
| **Ulti** | Usually play it straight, minimize the loss | you need one *specific* trick; big downside if you overreach |
| **Párti / 100** | Always squeeze for extra points | graded — every mistake-point is real GP |

## The mirror image (for the defender)

The same map tells you **when a soloist in trouble is dangerous**:
- Against a **betli** you think is beaten — *slow down*. This is exactly where they set traps and
  where one loose card throws it all away. Count every card.
- Against a **failing ulti/duri** — relax; once you hold the stopper, there's little they can do.
  Don't overthink it and don't get talked into a needless kontra just because they look nervous.

*(The engine's own "exploitation" research measures this directly: teaching a soloist to play the
trickiest safe line wins meaningful extra betlis against fallible defenders, but adds far less on
trick contracts — precisely because the trap headroom is where the map above says it is.)*
