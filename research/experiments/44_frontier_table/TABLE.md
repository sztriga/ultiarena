# Frontier self-play — 6000 deals, CHEAT-FREE bidder (exp44, 2026-08-02)

The exp29 table, re-measured after closing the talon leak. Every exp29 number was produced
by a bidder that decided whether to enter the auction while looking at the talon.

Blind pickup · full auction (any seat may open) · deployed play (exp31 exploit, exp36 betli
defense, terített pinning, PIMC, anti-tell mixer) · deployed per-unit kontra + rekontra ·
oracle scoring incl. silents. Bidder built from `ulti.bidding.frontier` — calibration ON,
exp37/39 betli heads ON. Seat 0 = forehand/opener.

`kontra%` = the CONTRACT itself was doubled. `any-k%` = any unit in the game was (a bid ulti
also exposes the párti, which defenders kontra ~80% of the time).

# exp44 frontier self-play (CHEAT-FREE bidder) — 6000 deals

| contract | count | freq | avg soloist GP | made% | kontra% | any-k% | avg /def | avg bids |
|---|---|---|---|---|---|---|---|---|
| ulti | 1269 | 21.1% | +8.49 | 93% | 2% | 75% | +4.24 | 1.07 |
| piros parti | 940 | 15.7% | +2.92 | 56% | 81% | 81% | +1.46 | 1.00 |
| piros ulti | 822 | 13.7% | +16.98 | 92% | 3% | 79% | +8.49 | 1.12 |
| ulti-40-100 | 122 | 2.0% | +15.79 | 96% | 1% | 1% | +7.89 | 1.11 |
| 40-100 | 118 | 2.0% | +5.92 | 86% | 0% | 0% | +2.96 | 1.01 |
| piros 40-100 | 116 | 1.9% | +11.55 | 85% | 0% | 0% | +5.78 | 1.03 |
| rebetli | 72 | 1.2% | +8.89 | 72% | 0% | 0% | +4.44 | 1.18 |
| piros ulti-40-100 | 51 | 0.8% | +25.65 | 86% | 2% | 2% | +12.82 | 1.27 |
| teritett betli | 24 | 0.4% | +40.00 | 100% | 0% | 0% | +20.00 | 1.17 |
| piros 20-100 | 24 | 0.4% | +13.00 | 71% | 0% | 0% | +6.50 | 1.00 |
| 20-100 | 20 | 0.3% | +8.60 | 75% | 0% | 0% | +4.30 | 1.00 |
| betli | 14 | 0.2% | +4.29 | 71% | 0% | 0% | +2.14 | 1.14 |
| 40-100-duri ≡ ulti-duri | 13 | 0.2% | +15.69 | 69% | 15% | 15% | +7.85 | 1.23 |
| piros 40-100-duri ≡ piros ulti-duri | 8 | 0.1% | +34.00 | 88% | 0% | 0% | +17.00 | 1.12 |
| duri | 6 | 0.1% | +0.00 | 50% | 0% | 0% | +0.00 | 1.00 |
| ulti-20-100 | 4 | 0.1% | +24.00 | 100% | 0% | 0% | +12.00 | 1.25 |
| piros ulti-20-100 | 3 | 0.1% | +52.00 | 100% | 0% | 0% | +26.00 | 1.00 |
| piros 20-100-duri | 3 | 0.1% | +26.67 | 67% | 100% | 100% | +13.33 | 1.33 |
| 20-100-duri | 1 | 0.0% | +28.00 | 100% | 0% | 0% | +14.00 | 1.00 |
| ulti-20-100-duri | 1 | 0.0% | +36.00 | 100% | 0% | 0% | +18.00 | 1.00 |
| piros ulti-20-100-duri | 1 | 0.0% | +72.00 | 100% | 0% | 0% | +36.00 | 2.00 |
| piros ulti-40-100-duri | 1 | 0.0% | +8.00 | 0% | 0% | 0% | +4.00 | 2.00 |
| ulti-40-100-duri | 1 | 0.0% | +28.00 | 100% | 0% | 0% | +14.00 | 1.00 |
| **passz** | 2366 | 39.4% | — | — | — | — | — | — |

## Overall
- deals: 6000 | played: 3634 (61%) | passz: 2366 (39%)
- soloist made 82% of played contracts
- mean soloist GP across played contracts: +9.87
- auction: avg 1.07 bids/played-deal; 7% contested

## Per-seat (seat 0 = forehand/opener)

| seat | mean GP/deal | won bid (soloist) | GP as soloist | GP as defender |
|---|---|---|---|---|
| P0 forehand | -0.783 | 1380 (23%) | +10.97 | -4.60 |
| P1 middle | +0.564 | 1192 (20%) | +9.28 | -5.08 |
| P2 rear | +0.220 | 1062 (18%) | +9.12 | -5.09 |

- zero-sum check: seat GP sums to +0.000 per deal (should be 0.000)

## Bleeding check

No contract loses GP on average.

## Versus exp29 (the leaky table)

| | exp29 (leaky) | exp44 (clean) |
|---|---|---|
| passz | 10% | **39%** |
| contested auctions | 44% | **7%** |
| avg bids / played deal | 1.62 | 1.07 |
| soloist made | 78% | 82% |
| mean soloist GP / played | +7.89 | +9.87 |
| contracts losing GP on average | **9** | **0** |
| piros parti | **−2.88** | **+2.92** |

Two readings, and both are true at once.

**The bidder got honest.** Nine contracts bled GP in exp29 — the terített-duri family worst
at −11 to −15. All nine are gone, and the whole ladder is now non-negative. `piros parti`
flipped from −2.88 to +2.92, which resolves the "piros parti loses money" item: the
marginal opener was exactly the blind-vs-informed decision, and it was being made while
looking at two cards it had not picked up.

**The bidder also got timid.** It passes 39% of deals against 10%, and contests only 7% of
auctions against 44%. Its mean GP per played contract went UP (+9.87 vs +7.89) precisely
because it now only plays hands it is sure of. That is not strength — it is selection. The
forehand pays −4 on every passz, which is why P0 sits at −0.783 GP/deal while the two seats
behind it are positive.

So the ladder is clean and the auction is asleep. exp45 measures what it costs.

## Per-seat

P0's −0.783 is the passz tax made visible: 39.4% of deals end with the forehand paying −4,
which is −1.58 GP/deal before it recovers anything as declarer. In exp29 (10% passz) the
same seat sat at +0.045.
