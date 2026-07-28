# Exp 13 — Defender 4-2-2-2 shape vs sol's ulti

## Question

When sol bids ulti (or holds the trump-7 with intent) and one defender
holds the classic "tank" shape — 4 trumps + 2-2-2 of the non-trump suits
— how often can sol still make ulti against god defense?

## Setup

- Sol always holds the trump-7 (precondition for ulti).
- One defender, chosen uniformly between D1 and D2, gets exactly:
  - 4 trumps drawn at random from {8, 9, J, Q, K, 10, A} (no 7)
  - 2 cards of each non-trump suit (10 cards total)
- Sol's other 9 cards are random from the remaining deck.
- Other defender + 2-card talon split the rest.
- God solver, perfect info, full alpha-beta. `contract='ulti'`.

## Results (N = 50,000)

| Cohort       | sol made ulti | sol failed | rate     |
|--------------|---------------|------------|----------|
| Overall      | 45            | 49,955     | **0.090%** |
| Tank = D1    | 20            | 24,972     | 0.080%   |
| Tank = D2    | 25            | 24,983     | 0.100%   |

## Interpretation

A defender holding any 4 of the top 7 trumps is essentially fatal: they
always hold a card that out-ranks the trump-7, they have enough trump
mass to survive 3 trump exchanges, and they have 2-2-2 on the side suits
so they can never be sluffed into a position where they must discard
trump. Sol's random other 9 cards almost never overcome this — only
~0.09% of deals produce a winnable ulti, and there is no positional
advantage from which defender is the tank (D1 ≈ D2 within sampling
noise).

## Perf note

50,000 full god solves in 6s wall on 4 workers ≈ 480 µs per solve.
Ulti is heavily pruned by the slim TT key, tight binary bounds [0, 10],
trump-7 reachability early-termination, and `_cull_ulti_*` cull rules.

## Realistic ulti-biased hands (v2)

v1 used a uniform sampler for sol's non-tank cards, so sol's hand was
weaker than what you'd see in a realistic ulti bid. v2 uses
`deal_ulti_biased(alpha=0.6)` (sol gets fat trump + biased side
strength, mandatory trump-7) and rejection-filters for the 4-2-2-2
tank shape. Accept rate ≈ 1.78%.

**Results (N = 5,116 accepted from 288,000 sampled, 5s wall on 4 workers):**

| Cohort       | sol made ulti | sol failed | rate     |
|--------------|---------------|------------|----------|
| Overall      | 182           | 4,934      | **3.56%** |
| Tank = D1    | 80            | 2,450      | 3.16%    |
| Tank = D2    | 102           | 2,484      | 3.94%    |

Sol's success rises sharply with how many trumps sol holds (the only
realistic counter to the tank):

| sol trumps | N      | made | rate   |
|------------|--------|------|--------|
| 2          | 1,395  | 0    | 0.00%  |
| 3          | 2,993  | 54   | 1.80%  |
| 4          | 728    | 128  | 17.58% |

**Takeaway**: realistic biased hands give sol ~40× more upside vs the
uniform v1 (0.09% → 3.56%), but the situation is still very bad — only
when sol piles on 4+ trumps does the rate cross 17%. The 4-2-2-2 tank
shape is a near-deterministic ulti killer in practice.

## Focus: sol-has-4-trumps (v3)

Drill-down on the only cohort with non-trivial success. Same setup as
v2 but rejection-filtered to also require sol holds exactly 4 trumps
(the trump-7 + 3 others). Combined accept rate ≈ 0.25%.

**Headline (N = 3,097 from 1.24M sampled, 9s wall):** sol makes ulti
in **18.11%** of these deals.

The setup is unusually tight: tank holds 4 of the 7 top trumps, sol
holds the 7 + 3 others = 4 more, so **sol + tank between them hold the
entire trump suit** (other_def and the talon are trump-void). The
trump battle is one-on-one between sol and tank, and ulti hinges on
who can drop the highest trump in trick 10.

Matrix of P(sol makes ulti) by top-sol-trump (excluding the 7) ×
top-tank-trump. Strongest trumps in the top-left:

| sol \ tank | ace   | 10    | king  | upper |
|-----------:|------:|------:|------:|------:|
| ace        | —     | 13.4% | 28.0% | 38.7% |
| 10         | 4.9%  | —     | —     | —     |
| king       | 0.0%  | —     | —     | —     |
| upper      | 0.0%  | —     | —     | —     |
| lower      | 0.0%  | —     | —     | —     |

Reading: the **only winnable region is the top row** — sol holds the
trump-ace. From there, success depends on how weak the tank's second-
in-command is: tank=upper (Jack) → 38.7%, king → 28.0%, 10 → 13.4%.
The 4.9% sliver at (sol=10, tank=ace) is flukes where sol's non-trump
structure forces the tank to dump the ace before trick 10. Every other
cell is 0% or empty.

The "—" cells are combinations that didn't appear in 3097 samples —
they are arithmetically narrow (e.g. sol top = king vs tank top = ace
requires sol's 3 trumps to be exactly {king, 9, 8}-style), so few
seeds produce them.

Equivalent view by sol's high-trump holdings (A=ace, 10, K=king):

| has A | has 10 | has K | N    | made | rate    |
|-------|--------|-------|------|------|---------|
| Y     | Y      | Y     | 506  | 196  | 38.74%  |
| Y     | Y      | .     | 778  | 218  | 28.02%  |
| Y     | .      | Y     | 530  | 66   | 12.45%  |
| Y     | .      | .     | 351  | 52   | 14.81%  |
| .     | Y      | K     | 338  | 24   | 7.10%   |
| .     | Y      | .     | 258  | 5    | 1.94%   |
| .     | .      | K     | 252  | 0    | 0.00%   |
| .     | .      | .     | 84   | 0    | 0.00%   |

## Reproduce

```
PYTHONPATH=. python3 experiments/13_def_4222_ulti/run_def_4222.py          # v1 uniform
PYTHONPATH=. python3 experiments/13_def_4222_ulti/run_def_4222_biased.py   # v2 ulti-biased
PYTHONPATH=. python3 experiments/13_def_4222_ulti/run_sol4_focus.py        # v3 sol-4-trumps drill-down
```
