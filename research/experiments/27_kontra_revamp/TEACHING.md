# Ulti — how often does a contract actually make?
*A study guide for defenders: when is it worth saying kontra?*

Every number below is **P(the soloist makes the contract)**, measured over thousands of real bid hands. Two regimes are shown side by side and never mixed:

- **Realistic** — against strong, human-like play (what you should expect at the table).
- **Perfect** — against flawless defense (the theoretical best a defender could do).

The two are usually close. Where **Perfect is much lower than Realistic**, the contract is *beatable but only with precise defense* — those are the hands worth studying.

---
## 1. The big picture

| Contract | Realistic make | Perfect make | Read |
|---|---|---|---|
| Párti | **36%** | 29% | a real coin-flip-ish fight — kontra often pays |
| Ulti | **83%** | 80% | very strong once bid — rarely worth kontra |
| 40-100 (négyszáz-száz) | **79%** | 74% | strong — rarely worth kontra |
| 20-100 (húsz-száz) | **62%** | 61% | beatable if you're trump-rich |
| Durchmars (duri) | **39%** | 36% | fragile — often beatable |
| Betli | **86%** | 66% | the defensive-skill contract (see §3) |

---
## 2. Trump count is (almost) everything for the trick contracts

For **ulti** and **durchmars**, the single biggest clue is how many trumps *you* (a defender) hold. Nothing else — not fancy win-probability math — beats just counting your trumps.


### Ulti
| Your trumps (as a defender) | hands | Realistic make | Perfect make |
|---|---|---|---|
| 1 | 90 | 99% | 100% |
| 2 | 1016 | 89% | 88% |
| 3 | 664 | 76% | 72% |
| 4 | 79 | 32% | 27% |

**Rule of thumb:** only kontra an ulti when you hold **4+ trumps** (then it fails ~2 times in 3). With 2–3 trumps, let it go.

### Durchmars (duri)
| Your trumps (as a defender) | hands | Realistic make | Perfect make |
|---|---|---|---|
| 0 | 129 | 50% | 44% |
| 2 | 24 | 58% | 58% |
| 3 | 40 | 2% | 5% |
| 4 | 12 | 0% | 0% |

**Rule of thumb:** a durchmars needs *every* trick. With **3+ trumps** you almost always have a stopper — kontra freely (it makes ~1 time in 20). With none, it still makes about half the time.

### 20-100 (húsz-száz)
| Your trumps (as a defender) | hands | Realistic make | Perfect make |
|---|---|---|---|
| 2 | 28 | 89% | 96% |
| 3 | 54 | 59% | 56% |
| 4 | 13 | 23% | 15% |

**Rule of thumb:** trump-rich (4+) makes it very beatable; otherwise the soloist usually gets there.

### 40-100 (négyszáz-száz)
| Your trumps (as a defender) | hands | Realistic make | Perfect make |
|---|---|---|---|
| 1 | 28 | 86% | 89% |
| 2 | 178 | 88% | 84% |
| 3 | 111 | 69% | 64% |
| 4 | 23 | 52% | 35% |

**Rule of thumb:** strong contract — even with 4 trumps it makes about half the time. Kontra only with real trump strength.

---
## 3. Betli — the contract that rewards *skill*

Betli (take zero tricks) is the one place perfect and realistic play diverge sharply: realistic make **86%**, but perfect defense holds the soloist to **66%**.

In other words: **a betli that *should* be beaten is only actually beaten by a defender who plays the squeeze correctly.** Unlike ulti or duri, you can't read betli off your trump count — there are no trumps. It's pure card-play skill, which is exactly why it's the contract most worth practicing on the defense.

---
## 4. Why over-eager kontra loses

A tempting mistake is to kontra whenever a contract *looks* hard. But the numbers say most bid contracts make: a soloist only bids what their hand supports. Ulti makes 83%, 40-100 makes 79%, betli 86%. Kontra doubles the stake **both ways** — so kontra-ing a contract that makes 80% of the time is a long-run loss. Save your kontra for the hands the tables above flag as genuinely beatable: párti (a genuine fight most hands), a **trump-rich** defense against ulti/duri, and betli when you can actually play the squeeze.

*(Data: several thousand champion-bid hands, played out both realistically (Monte-Carlo perfect-information search) and under a perfect-information solver. Sample sizes shown per row.)*
