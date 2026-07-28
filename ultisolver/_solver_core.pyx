# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# distutils: language = c
"""Fast alpha-beta endgame solver for Ulti (Cython implementation).

Supports pluggable contract evaluators via C-level function pointers.
The search engine (minimax, legal actions, apply/undo) is fully
contract-agnostic; the contract only affects terminal evaluation,
early termination, score bounds, and move ordering.

Supported contracts:
    "parti"       — soloist wins majority of card points
    "betli"       — soloist must take zero tricks
    "durchmars"   — soloist must win all 10 tricks
    "ulti"        — soloist must win last trick with trump 7 (binary)

Card encoding (matches the Python encoder):
    card_id = suit_idx * 8 + rank_idx  (0..31)
    suit: 0=HEARTS, 1=BELLS, 2=LEAVES, 3=ACORNS
    rank: 0=SEVEN, 1=EIGHT, 2=NINE, 3=JACK, 4=QUEEN, 5=KING, 6=TEN, 7=ACE

Usage:
    from ultisolver._solver_core import solve_root, solve_best

    values = solve_root(gs)                          # auto-detect contract
    values = solve_root(gs, contract="ulti")         # explicit contract
    card, val = solve_best(gs, contract="betli")
"""

from libc.stdlib cimport malloc, calloc, free

# ===========================================================================
#  Compile-time constants
# ===========================================================================

DEF C_NP = 3            # num players
DEF C_TRICKS = 10       # tricks per game
DEF C_LAST_BONUS = 10   # last trick bonus
DEF C_NO_TRUMP = -1     # sentinel for betli / no trump
DEF C_MAX_MOVES = 10    # max cards in a hand
DEF C_INF = 1000000.0   # +/- infinity for alpha-beta

# Contract IDs (used by _get_eval)
DEF EV_PARTI = 0
DEF EV_BETLI = 1
DEF EV_DURCHMARS = 2
DEF EV_ULTI = 4
DEF EV_MULTI = 5     # generic weighted-payoff evaluator (see _term_multi)

# Betli terminal values (binary: won all 0 tricks or not)
DEF C_BETLI_WIN = 1.0
DEF C_BETLI_LOSE = 0.0

# Durchmars terminal values (binary)
DEF C_DM_WIN = 1.0
DEF C_DM_LOSE = 0.0

# Ulti bonus/penalty (large enough to dominate card-point differences)
DEF C_ULTI_WIN = 1.0
DEF C_ULTI_LOSE = 0.0

# ===========================================================================
#  Inline bit helpers (portable, no compiler intrinsics)
# ===========================================================================

cdef inline int _popcount(unsigned int x) noexcept nogil:
    x = x - ((x >> 1) & 0x55555555u)
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u)
    return <int>((((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24)

cdef inline int _ctz(unsigned int x) noexcept nogil:
    cdef int n = 0
    if x == 0:
        return 32
    if (x & 0x0000FFFFu) == 0:
        n += 16; x >>= 16
    if (x & 0x000000FFu) == 0:
        n += 8; x >>= 8
    if (x & 0x0000000Fu) == 0:
        n += 4; x >>= 4
    if (x & 0x00000003u) == 0:
        n += 2; x >>= 2
    if (x & 0x00000001u) == 0:
        n += 1
    return n

# ===========================================================================
#  Card helpers
# ===========================================================================

cdef inline int _suit(int c) noexcept nogil:
    return c >> 3

cdef inline int _rank(int c) noexcept nogil:
    return c & 7

cdef inline unsigned int _bit(int c) noexcept nogil:
    return 1u << c

cdef inline unsigned int _smask(int s) noexcept nogil:
    return 0xFFu << (s * 8)

cdef inline int _pts(int c) noexcept nogil:
    cdef int r = c & 7
    if r >= 6:
        return 10
    return 0

cdef inline int _str_n(int c) noexcept nogil:
    return c & 7

cdef inline int _str_b(int c) noexcept nogil:
    cdef int r = c & 7
    if r == 3: return 4   # JACK
    if r == 4: return 5   # QUEEN
    if r == 5: return 6   # KING
    if r == 6: return 3   # TEN
    return r

cdef inline int _strength(int c, int betli) noexcept nogil:
    if betli:
        return _str_b(c)
    return _str_n(c)

# ===========================================================================
#  C-level game state
# ===========================================================================

cdef struct CState:
    unsigned int hands[3]
    int trump               # 0..3 or C_NO_TRUMP
    int betli
    int soloist
    int leader
    int trick_no
    int tc_n                # current trick card count (0, 1, 2)
    int tc_p[2]             # current trick players
    int tc_c[2]             # current trick card IDs
    int scores[3]
    int tricks_won[3]
    int has_ulti
    int binary              # 1 = binary-outcome contract (ulti/duri) → cull ignores pts
    # Last completed trick (for ulti check at terminal)
    int lt_p[3]             # last trick players
    int lt_c[3]             # last trick card IDs
    int lt_winner           # last trick winner (-1 if none)


cdef struct Undo:
    int player
    int card
    int done                # 1 = trick completed
    int stc_p[2]
    int stc_c[2]
    int stc_n
    int sleader
    int strick_no
    int sscores[3]
    int stw[3]
    # Saved last-trick info (only when done=1)
    int s_lt_p[3]
    int s_lt_c[3]
    int s_lt_winner


cdef struct Moves:
    int c[C_MAX_MOVES]
    int n

# ===========================================================================
#  Current player
# ===========================================================================

cdef inline int _cur(CState* s) noexcept nogil:
    cdef int p = s.leader
    cdef int i
    for i in range(s.tc_n):
        p = (p + 1) % C_NP
    return p

# ===========================================================================
#  Trick winner (unrolled for 3 cards)
# ===========================================================================

cdef inline int _tw3(
    int p0, int c0, int p1, int c1, int p2, int c2,
    int trump, int betli,
) noexcept nogil:
    cdef int led = _suit(c0)
    cdef int bp = p0, bc = c0, bt = (trump >= 0 and _suit(c0) == trump)
    cdef int ct, ss, bs

    ct = (trump >= 0 and _suit(c1) == trump)
    if ct and not bt:
        bp = p1; bc = c1; bt = 1
    elif ct and bt:
        ss = _strength(c1, betli); bs = _strength(bc, betli)
        if ss > bs:
            bp = p1; bc = c1
    elif not ct and not bt:
        if _suit(c1) == led:
            if _suit(bc) == led:
                ss = _strength(c1, betli); bs = _strength(bc, betli)
                if ss > bs:
                    bp = p1; bc = c1
            else:
                bp = p1; bc = c1

    ct = (trump >= 0 and _suit(c2) == trump)
    if ct and not bt:
        bp = p2; bc = c2; bt = 1
    elif ct and bt:
        ss = _strength(c2, betli); bs = _strength(bc, betli)
        if ss > bs:
            bp = p2; bc = c2
    elif not ct and not bt:
        if _suit(c2) == led:
            if _suit(bc) == led:
                ss = _strength(c2, betli); bs = _strength(bc, betli)
                if ss > bs:
                    bp = p2; bc = c2
            else:
                bp = p2; bc = c2

    return bp

# ===========================================================================
#  Apply / Undo
# ===========================================================================

cdef inline void _apply(CState* s, int card, Undo* u) noexcept nogil:
    cdef int p = _cur(s)
    cdef int w, pts, i

    u.player = p
    u.card = card
    s.hands[p] &= ~_bit(card)

    if s.tc_n < 2:
        s.tc_p[s.tc_n] = p
        s.tc_c[s.tc_n] = card
        s.tc_n += 1
        u.done = 0
        return

    # --- Trick completing (3rd card) ---
    u.done = 1
    # Save trick state
    u.stc_p[0] = s.tc_p[0]; u.stc_p[1] = s.tc_p[1]
    u.stc_c[0] = s.tc_c[0]; u.stc_c[1] = s.tc_c[1]
    u.stc_n = s.tc_n
    u.sleader = s.leader
    u.strick_no = s.trick_no
    for i in range(C_NP):
        u.sscores[i] = s.scores[i]
        u.stw[i] = s.tricks_won[i]
    # Save last-trick info
    u.s_lt_p[0] = s.lt_p[0]; u.s_lt_p[1] = s.lt_p[1]; u.s_lt_p[2] = s.lt_p[2]
    u.s_lt_c[0] = s.lt_c[0]; u.s_lt_c[1] = s.lt_c[1]; u.s_lt_c[2] = s.lt_c[2]
    u.s_lt_winner = s.lt_winner

    # Record this trick as the last trick (before clearing)
    s.lt_p[0] = s.tc_p[0]; s.lt_c[0] = s.tc_c[0]
    s.lt_p[1] = s.tc_p[1]; s.lt_c[1] = s.tc_c[1]
    s.lt_p[2] = p;          s.lt_c[2] = card

    # Resolve
    w = _tw3(s.tc_p[0], s.tc_c[0], s.tc_p[1], s.tc_c[1],
             p, card, s.trump, s.betli)
    pts = _pts(s.tc_c[0]) + _pts(s.tc_c[1]) + _pts(card)
    s.scores[w] += pts
    s.tricks_won[w] += 1
    s.trick_no += 1
    if s.trick_no == C_TRICKS:
        s.scores[w] += C_LAST_BONUS
    s.leader = w
    s.lt_winner = w
    s.tc_n = 0


cdef inline void _undo(CState* s, Undo* u) noexcept nogil:
    cdef int i
    s.hands[u.player] |= _bit(u.card)
    if u.done:
        s.tc_p[0] = u.stc_p[0]; s.tc_p[1] = u.stc_p[1]
        s.tc_c[0] = u.stc_c[0]; s.tc_c[1] = u.stc_c[1]
        s.tc_n = u.stc_n
        s.leader = u.sleader
        s.trick_no = u.strick_no
        for i in range(C_NP):
            s.scores[i] = u.sscores[i]
            s.tricks_won[i] = u.stw[i]
        s.lt_p[0] = u.s_lt_p[0]; s.lt_p[1] = u.s_lt_p[1]; s.lt_p[2] = u.s_lt_p[2]
        s.lt_c[0] = u.s_lt_c[0]; s.lt_c[1] = u.s_lt_c[1]; s.lt_c[2] = u.s_lt_c[2]
        s.lt_winner = u.s_lt_winner
    else:
        s.tc_n -= 1

# ===========================================================================
#  Legal actions (contract-agnostic — pure game mechanics)
# ===========================================================================

cdef inline int _max_str(unsigned int mask, int betli) noexcept nogil:
    cdef int best = -1, c, st
    while mask:
        c = _ctz(mask)
        st = _strength(c, betli)
        if st > best:
            best = st
        mask &= mask - 1
    return best


cdef void _legal(CState* s, Moves* m) noexcept nogil:
    cdef int p = _cur(s)
    cdef unsigned int hand = s.hands[p]
    cdef int n = 0
    cdef int led, c, i, j, mx, t7
    cdef unsigned int sc, ps, hi, tmp, trumps, tp, ht

    if s.tc_n == 0:
        tmp = hand
        while tmp:
            c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
    else:
        led = _suit(s.tc_c[0])
        sc = hand & _smask(led)

        if sc:
            if s.betli:
                # Must-beat with betli strength ordering
                ps = 0
                for i in range(s.tc_n):
                    if _suit(s.tc_c[i]) == led:
                        ps |= _bit(s.tc_c[i])
                if ps:
                    mx = _max_str(ps, 1)
                    hi = 0
                    tmp = sc
                    while tmp:
                        c = _ctz(tmp)
                        if _str_b(c) > mx:
                            hi |= _bit(c)
                        tmp &= tmp - 1
                    if hi:
                        tmp = hi
                        while tmp:
                            c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
                    else:
                        tmp = sc
                        while tmp:
                            c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
                else:
                    tmp = sc
                    while tmp:
                        c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
            else:
                ps = 0
                for i in range(s.tc_n):
                    if _suit(s.tc_c[i]) == led:
                        ps |= _bit(s.tc_c[i])
                if ps:
                    mx = _max_str(ps, 0)
                    hi = 0
                    tmp = sc
                    while tmp:
                        c = _ctz(tmp)
                        if _str_n(c) > mx:
                            hi |= _bit(c)
                        tmp &= tmp - 1
                    if hi:
                        tmp = hi
                        while tmp:
                            c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
                    else:
                        tmp = sc
                        while tmp:
                            c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
                else:
                    tmp = sc
                    while tmp:
                        c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
        else:
            if s.betli or s.trump < 0:
                tmp = hand
                while tmp:
                    c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
            else:
                trumps = hand & _smask(s.trump)
                if not trumps:
                    tmp = hand
                    while tmp:
                        c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
                else:
                    tp = 0
                    for i in range(s.tc_n):
                        if _suit(s.tc_c[i]) == s.trump:
                            tp |= _bit(s.tc_c[i])
                    if tp:
                        mx = _max_str(tp, 0)
                        ht = 0
                        tmp = trumps
                        while tmp:
                            c = _ctz(tmp)
                            if _str_n(c) > mx:
                                ht |= _bit(c)
                            tmp &= tmp - 1
                        if ht:
                            tmp = ht
                            while tmp:
                                c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
                        else:
                            tmp = trumps
                            while tmp:
                                c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
                    else:
                        tmp = trumps
                        while tmp:
                            c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1

    # 7esre tartás
    if (s.has_ulti and p == s.soloist and s.trump >= 0
            and s.trick_no < C_TRICKS - 1 and n > 1):
        t7 = s.trump * 8
        for i in range(n):
            if m.c[i] == t7:
                n -= 1
                for j in range(i, n):
                    m.c[j] = m.c[j + 1]
                break

    # Betli soloist: only keep highest-strength card per suit (dominant)
    cdef int best_str[4]
    cdef int best_card[4]
    cdef int si, bs, k
    if s.betli and p == s.soloist and n > 1:
        for si in range(4):
            best_str[si] = -1
            best_card[si] = -1
        for i in range(n):
            si = _suit(m.c[i])
            bs = _str_b(m.c[i])
            if bs > best_str[si]:
                best_str[si] = bs
                best_card[si] = m.c[i]
        # Compact: write saved card IDs back
        k = 0
        for si in range(4):
            if best_card[si] >= 0:
                m.c[k] = best_card[si]
                k += 1
        n = k

    m.n = n

# ===========================================================================
#  Remaining card points (for bounds calculation)
# ===========================================================================

cdef inline int _rem_pts(CState* s) noexcept nogil:
    cdef unsigned int all_c = s.hands[0] | s.hands[1] | s.hands[2]
    cdef int i
    for i in range(s.tc_n):
        all_c |= _bit(s.tc_c[i])
    cdef int pts = _popcount(all_c & 0xC0C0C0C0u) * 10
    if s.trick_no < C_TRICKS:
        pts += C_LAST_BONUS
    return pts

# ===========================================================================
#  Move ordering
# ===========================================================================

cdef void _order_default(Moves* m, CState* s, int maximising) noexcept nogil:
    """Standard ordering: trumps/high-strength for MAX, low for MIN."""
    cdef int i, j, n = m.n, ki, c, it, pt, st
    cdef int tmp
    cdef int keys[C_MAX_MOVES]

    if n <= 1:
        return

    for i in range(n):
        c = m.c[i]
        it = 1 if (s.trump >= 0 and _suit(c) == s.trump) else 0
        pt = _pts(c)
        st = _strength(c, s.betli)
        keys[i] = it * 10000 + pt * 100 + st

    for i in range(1, n):
        ki = keys[i]; tmp = m.c[i]; j = i - 1
        if maximising:
            while j >= 0 and keys[j] < ki:
                keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        else:
            while j >= 0 and keys[j] > ki:
                keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        keys[j + 1] = ki; m.c[j + 1] = tmp


cdef void _order_betli(Moves* m, CState* s, int maximising) noexcept nogil:
    """Betli-specific move ordering.

    Mirrors solvers/betli/solver.py:_move_key — sort by
    (suit_index, betli_strength, card_id) DESCENDING. This grouping
    "consider all of one suit first, biggest strength first" produces
    much better alpha-beta cutoffs than the generic ``_order_default``
    (which mixes suits and gives a 10/Ace point bonus that doesn't
    apply in betli). Same orientation for both MAX and MIN — the cull
    has already kept the soloist's dominant card per suit.
    """
    cdef int n = m.n
    cdef int i, j, c, tmp
    cdef int keys[C_MAX_MOVES]
    cdef int ki

    if n <= 1:
        return

    for i in range(n):
        c = m.c[i]
        # Composite descending key: (suit << 12) | (betli_strength << 6) | (card_id & 0x3F)
        # All three tiers fit; lexicographic on the packed int = lexicographic on the tuple.
        keys[i] = (_suit(c) << 12) | (_str_b(c) << 6) | (c & 0x3F)

    for i in range(1, n):
        ki = keys[i]; tmp = m.c[i]; j = i - 1
        while j >= 0 and keys[j] < ki:
            keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        keys[j + 1] = ki; m.c[j + 1] = tmp


# ---------------------------------------------------------------------------
#  Parti experimental orderings (for speed benchmark — see set_parti_order)
# ---------------------------------------------------------------------------

cdef void _order_parti_pts_first(Moves* m, CState* s, int maximising) noexcept nogil:
    """Points-first: pt * 1000 + it * 100 + st. Surfaces high-value captures
    before raw trump/strength."""
    cdef int i, j, n = m.n, ki, c, it, pt, st, tmp
    cdef int keys[C_MAX_MOVES]
    if n <= 1:
        return
    for i in range(n):
        c = m.c[i]
        it = 1 if (s.trump >= 0 and _suit(c) == s.trump) else 0
        pt = _pts(c)
        st = _strength(c, s.betli)
        keys[i] = pt * 1000 + it * 100 + st
    for i in range(1, n):
        ki = keys[i]; tmp = m.c[i]; j = i - 1
        if maximising:
            while j >= 0 and keys[j] < ki:
                keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        else:
            while j >= 0 and keys[j] > ki:
                keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        keys[j + 1] = ki; m.c[j + 1] = tmp


cdef void _order_parti_winner_strength(Moves* m, CState* s, int maximising) noexcept nogil:
    """Composite winner-strength then points: ((it<<3)|st)*100 + pt.
    Cards more likely to win the trick (incl. trump bonus baked in) come
    first, points as tiebreak."""
    cdef int i, j, n = m.n, ki, c, it, pt, st, tmp
    cdef int keys[C_MAX_MOVES]
    if n <= 1:
        return
    for i in range(n):
        c = m.c[i]
        it = 1 if (s.trump >= 0 and _suit(c) == s.trump) else 0
        pt = _pts(c)
        st = _strength(c, s.betli)
        keys[i] = ((it << 3) | st) * 100 + pt
    for i in range(1, n):
        ki = keys[i]; tmp = m.c[i]; j = i - 1
        if maximising:
            while j >= 0 and keys[j] < ki:
                keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        else:
            while j >= 0 and keys[j] > ki:
                keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        keys[j + 1] = ki; m.c[j + 1] = tmp


# Global selector: 0=legacy default (it,pt,st), 1=pts_first (now canonical),
# 2=winner_strength. Set via set_parti_order(i) from Python; read by _get_eval.
# Default = 1 (pts_first) — benchmarked at 3× speedup over the legacy
# ordering with bit-identical values; see experiments/06_parti_value_net/
# move_order_experiment.md.
cdef int _g_parti_order_id = 1

def set_parti_order(int order_id):
    """Swap the parti move-ordering function. 0=default, 1=pts_first,
    2=winner_strength. Affects EV_PARTI only."""
    global _g_parti_order_id
    if order_id < 0 or order_id > 2:
        raise ValueError(f"order_id must be 0/1/2, got {order_id}")
    _g_parti_order_id = order_id

def get_parti_order():
    return _g_parti_order_id


# ---------------------------------------------------------------------------
#  Ulti-specific orderings (benchmarked via set_ulti_order)
# ---------------------------------------------------------------------------
# Points don't affect the binary ulti outcome, so we drop the pt term from
# the sort key. Trump still ranks above non-trump because trick ownership
# matters for ulti (whoever has the highest trump in trick 10 wins).

cdef void _order_ulti_no_pts(Moves* m, CState* s, int maximising) noexcept nogil:
    """Key = is_trump*100 + strength. Trump-first, then high-strength;
    no point-bonus distortion."""
    cdef int i, j, n = m.n, ki, c, it, st, tmp
    cdef int keys[C_MAX_MOVES]
    if n <= 1:
        return
    for i in range(n):
        c = m.c[i]
        it = 1 if (s.trump >= 0 and _suit(c) == s.trump) else 0
        st = _strength(c, s.betli)
        keys[i] = it * 100 + st
    for i in range(1, n):
        ki = keys[i]; tmp = m.c[i]; j = i - 1
        if maximising:
            while j >= 0 and keys[j] < ki:
                keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        else:
            while j >= 0 and keys[j] > ki:
                keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        keys[j + 1] = ki; m.c[j + 1] = tmp


cdef void _order_ulti_pull_duck(Moves* m, CState* s, int maximising) noexcept nogil:
    """Asymmetric: trumps ranked HIGH-to-LOW for MAX, non-trumps ranked
    LOW-to-HIGH for MAX (dump junk first to save high cards).
    Key = is_trump*1000 + (is_trump ? strength : (10 - strength))."""
    cdef int i, j, n = m.n, ki, c, it, st, tmp
    cdef int keys[C_MAX_MOVES]
    if n <= 1:
        return
    for i in range(n):
        c = m.c[i]
        it = 1 if (s.trump >= 0 and _suit(c) == s.trump) else 0
        st = _strength(c, s.betli)
        if it:
            keys[i] = it * 1000 + st
        else:
            keys[i] = (10 - st)
    for i in range(1, n):
        ki = keys[i]; tmp = m.c[i]; j = i - 1
        if maximising:
            while j >= 0 and keys[j] < ki:
                keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        else:
            while j >= 0 and keys[j] > ki:
                keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        keys[j + 1] = ki; m.c[j + 1] = tmp


cdef void _order_ulti_t7_first(Moves* m, CState* s, int maximising) noexcept nogil:
    """Like no_pts but when the soloist's trump-7 is legal (only on trick 9
    in non-forced play under 7esre tartás), try it first. This is the move
    that resolves the ulti question for MAX; testing it first prunes the
    other trick-10 leads aggressively. For MIN the bonus is reversed."""
    cdef int i, j, n = m.n, ki, c, it, st, tmp, t7
    cdef int keys[C_MAX_MOVES]
    if n <= 1:
        return
    t7 = s.trump * 8 if s.trump >= 0 else -1
    for i in range(n):
        c = m.c[i]
        it = 1 if (s.trump >= 0 and _suit(c) == s.trump) else 0
        st = _strength(c, s.betli)
        keys[i] = it * 100 + st
        if c == t7:
            keys[i] += 10000   # jump to top for MAX (or bottom under inversion)
    for i in range(1, n):
        ki = keys[i]; tmp = m.c[i]; j = i - 1
        if maximising:
            while j >= 0 and keys[j] < ki:
                keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        else:
            while j >= 0 and keys[j] > ki:
                keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        keys[j + 1] = ki; m.c[j + 1] = tmp


cdef void _order_ulti_human(Moves* m, CState* s, int maximising) noexcept nogil:
    """Lead-high / follow-low heuristic for both sides.

    Idea: the *true* optimal move when following is usually the cheapest
    card that wins (ducking principle), and when leading it's typically
    the highest card. Alpha-beta cutoffs fire fastest when the first move
    tested is the actual optimum.

    Key = is_trump * 100 + strength; direction depends on leading/following
    and side. MAX leading and MIN leading: descending. MAX following and
    MIN following: ascending."""
    cdef int i, j, n = m.n, ki, c, it, st, tmp
    cdef int keys[C_MAX_MOVES]
    cdef int desc
    if n <= 1:
        return
    if s.tc_n == 0:
        desc = 1   # leading → high-first regardless of side
    else:
        desc = 0   # following → low-first regardless of side
    for i in range(n):
        c = m.c[i]
        it = 1 if (s.trump >= 0 and _suit(c) == s.trump) else 0
        st = _strength(c, s.betli)
        keys[i] = it * 100 + st
    for i in range(1, n):
        ki = keys[i]; tmp = m.c[i]; j = i - 1
        if desc:
            while j >= 0 and keys[j] < ki:
                keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        else:
            while j >= 0 and keys[j] > ki:
                keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        keys[j + 1] = ki; m.c[j + 1] = tmp


# Global selector for EV_ULTI ordering.
# 0 = _order_default (baseline), 1 = no_pts, 2 = pull_duck, 3 = t7_first,
# 4 = human (lead-high / follow-low for both sides)
cdef int _g_ulti_order_id = 1

def set_ulti_order(int order_id):
    """Swap the EV_ULTI move-ordering function.
    0=default, 1=no_pts, 2=pull_duck, 3=t7_first, 4=human."""
    global _g_ulti_order_id
    if order_id < 0 or order_id > 4:
        raise ValueError(f"order_id must be 0..4, got {order_id}")
    _g_ulti_order_id = order_id

def get_ulti_order():
    return _g_ulti_order_id


# Global toggle for the úr-vagyok proven-safe predicate (1=on, 0=off).
cdef int _g_ulti_proven_safe = 1

def set_ulti_proven_safe(int enabled):
    """Toggle the úr-vagyok early-WIN predicate for EV_ULTI."""
    global _g_ulti_proven_safe
    _g_ulti_proven_safe = 1 if enabled else 0

def get_ulti_proven_safe():
    return _g_ulti_proven_safe


# ===========================================================================
#  Contract evaluator (Strategy pattern via C function pointers)
# ===========================================================================

ctypedef float (*TermFn)(CState*) noexcept nogil
ctypedef int   (*EarlyFn)(CState*) noexcept nogil
ctypedef void  (*BoundsFn)(CState*, float*, float*) noexcept nogil
ctypedef void  (*OrderFn)(Moves*, CState*, int) noexcept nogil
ctypedef void  (*CullFn)(Moves*, CState*, int) noexcept nogil
ctypedef unsigned long long (*TTExtraFn)(CState*) noexcept nogil

cdef struct ContractEval:
    TermFn    terminal     # score at terminal / early-terminal
    EarlyFn   early_term   # can we stop before trick 10?
    BoundsFn  bounds       # [lo, hi] for futility pruning
    OrderFn   order        # move ordering heuristic
    CullFn    cull         # drop strictly-dominated moves before search
    TTExtraFn tt_extra     # pack state into TT key (contract-specific)
    int       use_tt       # 1 → look up / store in transposition table

# ---------------------------------------------------------------------------
#  Default cull (no-op)
# ---------------------------------------------------------------------------

cdef void _cull_noop(Moves* m, CState* s, int maximising) noexcept nogil:
    pass

# ---------------------------------------------------------------------------
#  Betli soloist suit-dominance cull
#
# When the soloist is on move and has multiple cards of the same suit,
# only the strongest (in betli ranking) is dominant — playing a weaker
# card of the same suit can only delay forced wins, never avoid them
# (proof: lifted from solvers/betli/solver.py and trickster's pure-Python
# fallback, both of which apply this cull). Collapses soloist branching
# from 3-4 down to ~1 in most positions.
# ---------------------------------------------------------------------------

cdef void _cull_betli_dominant(Moves* m, CState* s, int maximising) noexcept nogil:
    if m.n <= 1:
        return
    if maximising:
        _cull_betli_soloist(m, s)
    else:
        _cull_betli_def_groups(m, s)


cdef void _cull_betli_soloist(Moves* m, CState* s) noexcept nogil:
    """Suit-dominance: for the soloist, only the strongest card per suit
    needs to be considered. Strict dominance — playing a weaker card of
    the same suit can only delay forced wins, never avoid them."""
    cdef int per_suit_card[4]
    cdef int per_suit_str [4]
    cdef int i, c, su, st

    for i in range(4):
        per_suit_card[i] = -1
        per_suit_str [i] = -1

    for i in range(m.n):
        c  = m.c[i]
        su = _suit(c)
        st = _str_b(c)
        if st > per_suit_str[su]:
            per_suit_str [su] = st
            per_suit_card[su] = c

    cdef int new_n = 0
    for i in range(4):
        if per_suit_card[i] >= 0:
            m.c[new_n] = per_suit_card[i]
            new_n += 1

    m.n = new_n


cdef void _cull_betli_def_groups(Moves* m, CState* s) noexcept nogil:
    """Equivalence-group cull for defender moves.

    Two cards A, B of the same suit in defender's hand are interchangeable
    iff no card of that suit with strength strictly between A and B exists
    in any *other* player's hand or in the current trick. Cards captured
    or in the talon are out of circulation and do not break equivalence
    (matches Milan's 'plugged hole' intuition). Within each equivalence
    group only one representative needs to be searched; we keep the
    highest-strength card (consistent with the soloist convention).
    """
    cdef int n = m.n
    cdef int cur = _cur(s)

    # 1. Build per-suit bitmap of strengths held by *other* live cards.
    cdef int other_str[4]
    cdef int suit, p, i, k, c
    cdef unsigned int h

    for suit in range(4):
        other_str[suit] = 0

    for p in range(C_NP):
        if p == cur:
            continue
        h = s.hands[p]
        while h != 0:
            c = _ctz(h)
            other_str[_suit(c)] |= (1 << _str_b(c))
            h &= h - 1

    for i in range(s.tc_n):
        c = s.tc_c[i]
        other_str[_suit(c)] |= (1 << _str_b(c))

    # 2. Group m.c[] cards by suit and sort each group by strength ascending.
    cdef int idx_by_suit[4][8]
    cdef int str_by_suit[4][8]
    cdef int n_by_suit[4]

    for suit in range(4):
        n_by_suit[suit] = 0

    for i in range(n):
        c = m.c[i]
        suit = _suit(c)
        idx_by_suit[suit][n_by_suit[suit]] = i
        str_by_suit[suit][n_by_suit[suit]] = _str_b(c)
        n_by_suit[suit] += 1

    cdef int j, tmp_str, tmp_idx
    for suit in range(4):
        for i in range(1, n_by_suit[suit]):
            tmp_str = str_by_suit[suit][i]
            tmp_idx = idx_by_suit[suit][i]
            j = i - 1
            while j >= 0 and str_by_suit[suit][j] > tmp_str:
                str_by_suit[suit][j + 1] = str_by_suit[suit][j]
                idx_by_suit[suit][j + 1] = idx_by_suit[suit][j]
                j -= 1
            str_by_suit[suit][j + 1] = tmp_str
            idx_by_suit[suit][j + 1] = tmp_idx

    # 3. Walk each suit ascending; an "other" strength strictly between two
    #    consecutive of our strengths ends the current equivalence group.
    cdef int keep[C_MAX_MOVES]
    for i in range(n):
        keep[i] = 0

    cdef int prev_str, cur_str, has_between, group_top_idx
    for suit in range(4):
        if n_by_suit[suit] == 0:
            continue
        group_top_idx = idx_by_suit[suit][0]
        prev_str = str_by_suit[suit][0]
        for i in range(1, n_by_suit[suit]):
            cur_str = str_by_suit[suit][i]
            has_between = 0
            for k in range(prev_str + 1, cur_str):
                if other_str[suit] & (1 << k):
                    has_between = 1
                    break
            if has_between:
                keep[group_top_idx] = 1
                group_top_idx = idx_by_suit[suit][i]
            else:
                group_top_idx = idx_by_suit[suit][i]
            prev_str = cur_str
        keep[group_top_idx] = 1

    # 4. Compact m.c[] in-place, preserving order.
    cdef int new_n = 0
    for i in range(n):
        if keep[i]:
            m.c[new_n] = m.c[i]
            new_n += 1
    m.n = new_n

# ---------------------------------------------------------------------------
#  Parti block-equivalence cull
#
# Two cards A, B of the same suit in the current mover's hand are
# interchangeable iff
#   (1) no card of that suit with parti-strength strictly between A and B
#       exists in any *other* player's hand or in the current trick, AND
#   (2) A and B have the same card-point value (both 0pts or both 10pts).
# Condition (2) is what distinguishes parti from betli: J,Q,K is one
# block (all 0pts) but K → 10 breaks the block (0pts → 10pts) even
# though they're rank-adjacent in parti order. Within a block we keep
# the highest-strength representative.
# ---------------------------------------------------------------------------

cdef void _cull_parti_blocks(Moves* m, CState* s, int maximising) noexcept nogil:
    cdef int n = m.n
    if n <= 1:
        return
    cdef int cur = _cur(s)

    # Ulti special case: when the soloist is the mover in a has_ulti
    # game, the trump-7 carries the binary ulti payoff and is not
    # equivalent to any neighbouring trump even if no opponent card
    # sits between them. We tag its pts column with a sentinel (-1)
    # so the block-splitter (cur_pts != prev_pts) always isolates it.
    # 7esre tartás already removes the trump-7 from m on tricks 0-8
    # (cf. _legal lines 415-424), so this guard only fires on trick 9.
    #
    # Multi case: when EV_MULTI is active with a non-zero silent_ulti
    # weight, the trump-7 carries the silent ulti payoff and must stay
    # isolated for *whoever* holds it (both sol and def can silent-ulti
    # or bukott). Same sentinel mechanism, just a broader gate.
    cdef int t7 = -1
    if s.trump >= 0:
        if s.has_ulti and cur == s.soloist:
            t7 = s.trump * 8
        elif _g_multi_w_silent_ulti != 0.0:
            t7 = s.trump * 8

    cdef int other_str[4]
    cdef int suit, p, i, k, c
    cdef unsigned int h

    for suit in range(4):
        other_str[suit] = 0

    for p in range(C_NP):
        if p == cur:
            continue
        h = s.hands[p]
        while h != 0:
            c = _ctz(h)
            other_str[_suit(c)] |= (1 << _strength(c, s.betli))
            h &= h - 1

    for i in range(s.tc_n):
        c = s.tc_c[i]
        other_str[_suit(c)] |= (1 << _strength(c, s.betli))

    cdef int idx_by_suit[4][8]
    cdef int str_by_suit[4][8]
    cdef int pts_by_suit[4][8]
    cdef int n_by_suit[4]

    for suit in range(4):
        n_by_suit[suit] = 0

    # In binary-outcome contracts (ulti, duri) points are irrelevant to
    # the value — treat every card as point-equivalent (pts=0) so 10s
    # and Aces don't gratuitously split blocks. The trump-7 keeps its
    # -1 sentinel in ulti to stay isolated (duri has has_ulti=0 so t7
    # is -1-unreachable).
    cdef int binary = s.binary

    for i in range(n):
        c = m.c[i]
        suit = _suit(c)
        idx_by_suit[suit][n_by_suit[suit]] = i
        str_by_suit[suit][n_by_suit[suit]] = _strength(c, s.betli)
        if c == t7:
            pts_by_suit[suit][n_by_suit[suit]] = -1   # ulti bonus card; non-equivalent
        elif binary:
            pts_by_suit[suit][n_by_suit[suit]] = 0    # binary outcome ignores points
        else:
            pts_by_suit[suit][n_by_suit[suit]] = _pts(c)
        n_by_suit[suit] += 1

    cdef int j, tmp_str, tmp_idx, tmp_pts
    for suit in range(4):
        for i in range(1, n_by_suit[suit]):
            tmp_str = str_by_suit[suit][i]
            tmp_idx = idx_by_suit[suit][i]
            tmp_pts = pts_by_suit[suit][i]
            j = i - 1
            while j >= 0 and str_by_suit[suit][j] > tmp_str:
                str_by_suit[suit][j + 1] = str_by_suit[suit][j]
                idx_by_suit[suit][j + 1] = idx_by_suit[suit][j]
                pts_by_suit[suit][j + 1] = pts_by_suit[suit][j]
                j -= 1
            str_by_suit[suit][j + 1] = tmp_str
            idx_by_suit[suit][j + 1] = tmp_idx
            pts_by_suit[suit][j + 1] = tmp_pts

    cdef int keep[C_MAX_MOVES]
    for i in range(n):
        keep[i] = 0

    cdef int prev_str, cur_str, prev_pts, cur_pts, has_between, group_top_idx
    for suit in range(4):
        if n_by_suit[suit] == 0:
            continue
        group_top_idx = idx_by_suit[suit][0]
        prev_str = str_by_suit[suit][0]
        prev_pts = pts_by_suit[suit][0]
        for i in range(1, n_by_suit[suit]):
            cur_str = str_by_suit[suit][i]
            cur_pts = pts_by_suit[suit][i]
            has_between = 0
            for k in range(prev_str + 1, cur_str):
                if other_str[suit] & (1 << k):
                    has_between = 1
                    break
            if has_between or cur_pts != prev_pts:
                keep[group_top_idx] = 1
            group_top_idx = idx_by_suit[suit][i]
            prev_str = cur_str
            prev_pts = cur_pts
        keep[group_top_idx] = 1

    cdef int new_n = 0
    for i in range(n):
        if keep[i]:
            m.c[new_n] = m.c[i]
            new_n += 1
    m.n = new_n

# ---------------------------------------------------------------------------
#  Parti evaluator (card points)
# ---------------------------------------------------------------------------

cdef inline int _def_score(CState* s) noexcept nogil:
    cdef int dp = 0
    cdef int i
    for i in range(C_NP):
        if i != s.soloist:
            dp += s.scores[i]
    return dp

# Parti scoring mode: 0=binary win/loss (new default), 1=margin/sol_pts (legacy).
cdef int _g_parti_mode = 0

def set_parti_mode(int mode):
    """0=binary (sol_pts > def_pts → ±1), 1=margin (sol's absolute pts)."""
    global _g_parti_mode
    if mode not in (0, 1):
        raise ValueError(f"mode must be 0 or 1, got {mode}")
    _g_parti_mode = mode

def get_parti_mode():
    return _g_parti_mode

cdef float _term_parti(CState* s) noexcept nogil:
    # Binary win/loss: sol's total points (incl. declared marriages) vs
    # the defenders' combined total. Sol wins parti iff strictly greater.
    # Scale 0/1 to match EV_ULTI / EV_BETLI / EV_DURCHMARS — uniform
    # binary contract scale across the codebase.
    if _g_parti_mode == 1:
        return <float>s.scores[s.soloist]
    return 1.0 if s.scores[s.soloist] > _def_score(s) else 0.0

cdef int _early_parti(CState* s) noexcept nogil:
    return 0

cdef void _bounds_parti(CState* s, float* lo, float* hi) noexcept nogil:
    cdef int sp = s.scores[s.soloist]
    cdef int dp
    cdef int rem = _rem_pts(s)
    if _g_parti_mode == 1:
        lo[0] = <float>sp
        hi[0] = <float>(sp + rem)
        return
    # Binary bounds: collapse to a single value once the outcome is forced.
    dp = _def_score(s)
    if sp > dp + rem:
        lo[0] = 1.0; hi[0] = 1.0
    elif sp + rem <= dp:
        lo[0] = 0.0; hi[0] = 0.0
    else:
        lo[0] = 0.0; hi[0] = 1.0

# ---------------------------------------------------------------------------
#  Betli evaluator (binary: 0 tricks taken = win)
# ---------------------------------------------------------------------------

cdef float _term_betli(CState* s) noexcept nogil:
    """Betli terminal: soloist takes 0 tricks → WIN (C_TRICKS), else
    LOSE (0). For losing terminals we add a tiny ``trick_no / (TRICKS*10)``
    tie-breaker (mirrors solvers/betli/solver.py:_terminal_value), so
    earlier forced losses sort below later ones — the defenders' MIN
    naturally surfaces the shortest forced loss first, tightening
    alpha-beta bounds and giving the TT cleaner EXACT classifications.
    The win case (sol_tricks=0) plays the full game so no tie-break needed.
    """
    if s.tricks_won[s.soloist] == 0:
        return C_BETLI_WIN
    # Tie-breaker is < 1 so it never crosses an integer boundary.
    return C_BETLI_LOSE + (<float>s.trick_no) / (<float>(C_TRICKS * 10))

cdef int _soloist_betli_proven_safe(CState* s) noexcept nogil:
    """Milan's no-holes predicate: in every suit soloist holds cards, every
    live opponent card outranks soloist's highest card in that suit.

    Why this is sound for betli (with soloist only allowed to lead trick 1):
    * Soloist leads at most one trick (the first). After that, only a defender
      can ever lead, and only suits *they* still hold can be led.
    * In any trick where suit S is led, defenders must follow with S if they
      have it. Their S cards are all strictly higher than soloist's max in S
      (predicate), so any defender forced-follow card beats soloist's card.
      If a defender is void in S they discard, but at least one defender plays
      an S card higher than soloist's — defender wins.
    * Defenders cannot lead S to lose on purpose, because their lowest S card
      is still higher than soloist's max. They can choose discards to deplete
      their S cards, but once both defenders are void, S can no longer be led;
      soloist's remaining S cards are forced into harmless discards (not in
      led suit → never win the trick).
    * "Live" excludes captured tricks and the talon (out of play), so a "hole"
      filled by a captured/talon card doesn't break the safety.

    Conservative: only fires at a trick boundary (``tc_n == 0``) to avoid
    reasoning about partial tricks.
    """
    if s.tc_n != 0:
        return 0
    cdef int sol = s.soloist
    cdef unsigned int sol_hand = s.hands[sol]
    cdef unsigned int opp_hand = 0
    cdef int p, suit, c, str_val
    cdef unsigned int h

    for p in range(C_NP):
        if p != sol:
            opp_hand |= s.hands[p]

    cdef int sol_max[4]
    cdef int opp_min[4]
    cdef int has_sol[4]

    for suit in range(4):
        sol_max[suit] = -1
        opp_min[suit] = 99
        has_sol[suit] = 0

    h = sol_hand
    while h != 0:
        c = _ctz(h)
        suit = _suit(c)
        str_val = _str_b(c)
        if str_val > sol_max[suit]:
            sol_max[suit] = str_val
        has_sol[suit] = 1
        h &= h - 1

    h = opp_hand
    while h != 0:
        c = _ctz(h)
        suit = _suit(c)
        str_val = _str_b(c)
        if str_val < opp_min[suit]:
            opp_min[suit] = str_val
        h &= h - 1

    for suit in range(4):
        if has_sol[suit] and sol_max[suit] >= opp_min[suit]:
            return 0
    return 1


cdef int _early_betli(CState* s) noexcept nogil:
    if s.tricks_won[s.soloist] > 0:
        return 1
    return _soloist_betli_proven_safe(s)

cdef void _bounds_betli(CState* s, float* lo, float* hi) noexcept nogil:
    # Only called when soloist has 0 tricks (early_term handles the rest)
    lo[0] = C_BETLI_LOSE
    hi[0] = C_BETLI_WIN

# ---------------------------------------------------------------------------
#  Durchmars evaluator (binary: all 10 tricks = win)
# ---------------------------------------------------------------------------

# Toggle for the "úr vagyok" early-WIN predicate (1=on, 0=off).
cdef int _g_dm_proven_safe = 1

def set_dm_proven_safe(int enabled):
    """Toggle the úr-vagyok early-WIN predicate for EV_DURCHMARS."""
    global _g_dm_proven_safe
    _g_dm_proven_safe = 1 if enabled else 0

def get_dm_proven_safe():
    return _g_dm_proven_safe


cdef int _soloist_dm_proven_safe(CState* s) noexcept nogil:
    """Úr vagyok for durchmars: soloist on lead at a trick boundary and can
    guarantee winning every remaining trick.

    Conditions (using ``_strength(c, s.betli)`` so the predicate is correct
    in both colored and colorless modes):

      (1) tc_n == 0 (trick boundary)
      (2) current player is sol (sol leads)
      (3) for every NON-TRUMP suit S where sol holds cards,
            sol_min(S) > opp_max(S)
          (sol can clear opps' S-cards one at a time; opps voids just
          discard since they never get to lead in a winning duri)
      (4) colored mode (trump >= 0) — trump must be tame:
          either opps hold zero trumps,
          OR  sol_min(trump) > opp_max(trump)  AND  sol holds at least as
              many trumps as opps combined (so sol can lead trump first
              and drain opps' trumps before touching any non-trump suit
              where opps are void)
          colorless mode (trump < 0): trump clause vacuously satisfied.

    Soundness sketch: under (3)+(4) sol drains opps' trumps with her own
    trumps first (each lead wins by sol_min(t) > opp_max(t)); then on
    every remaining suit lead, opps either follow with a losing card
    (sol_min(S) > opp_max(S)) or discard (they have no trump left to
    ruff). Sol wins every trick.
    """
    if s.tc_n != 0:
        return 0
    cdef int sol = s.soloist
    if _cur(s) != sol:
        return 0

    cdef unsigned int sol_hand = s.hands[sol]
    cdef unsigned int opp_hand = 0
    cdef int p, suit, c, str_val
    cdef unsigned int h
    cdef int trump_suit = s.trump

    for p in range(C_NP):
        if p != sol:
            opp_hand |= s.hands[p]

    cdef int sol_min[4]
    cdef int opp_max[4]
    cdef int has_sol[4]
    cdef int sol_count_t = 0
    cdef int opp_count_t = 0
    cdef int suit_i
    for suit_i in range(4):
        sol_min[suit_i] = 99
        opp_max[suit_i] = -1
        has_sol[suit_i] = 0

    h = sol_hand
    while h != 0:
        c = _ctz(h)
        suit = _suit(c)
        str_val = _strength(c, s.betli)
        if str_val < sol_min[suit]:
            sol_min[suit] = str_val
        has_sol[suit] = 1
        if trump_suit >= 0 and suit == trump_suit:
            sol_count_t += 1
        h &= h - 1

    h = opp_hand
    while h != 0:
        c = _ctz(h)
        suit = _suit(c)
        str_val = _strength(c, s.betli)
        if str_val > opp_max[suit]:
            opp_max[suit] = str_val
        if trump_suit >= 0 and suit == trump_suit:
            opp_count_t += 1
        h &= h - 1

    # Non-trump suits sol holds: must strictly dominate.
    for suit_i in range(4):
        if trump_suit >= 0 and suit_i == trump_suit:
            continue
        if has_sol[suit_i] and sol_min[suit_i] <= opp_max[suit_i]:
            return 0

    # Trump (colored only) — must be either absent on opp side or tame.
    if trump_suit >= 0 and opp_count_t > 0:
        if not has_sol[trump_suit]:
            return 0  # opps have trump, sol doesn't → opps ruff anything
        if sol_min[trump_suit] <= opp_max[trump_suit]:
            return 0
        if sol_count_t < opp_count_t:
            return 0  # can't drain opp trumps before having to lead non-trump
    return 1


cdef int _early_dm_outcome(CState* s) noexcept nogil:
    """Returns +1 = early WIN, -1 = early LOSS, 0 = undecided."""
    if s.trick_no > s.tricks_won[s.soloist]:
        return -1
    if _g_dm_proven_safe and _soloist_dm_proven_safe(s):
        return 1
    return 0


cdef float _term_dm(CState* s) noexcept nogil:
    if s.trick_no >= C_TRICKS:
        if s.tricks_won[s.soloist] == C_TRICKS:
            return C_DM_WIN
        return C_DM_LOSE
    if _early_dm_outcome(s) > 0:
        return C_DM_WIN
    return C_DM_LOSE

cdef int _early_dm(CState* s) noexcept nogil:
    return _early_dm_outcome(s) != 0

cdef void _bounds_dm(CState* s, float* lo, float* hi) noexcept nogil:
    lo[0] = C_DM_LOSE
    hi[0] = C_DM_WIN

# ---------------------------------------------------------------------------
#  Parti + Ulti evaluator (card points + ulti bonus/penalty)
# ---------------------------------------------------------------------------

cdef inline int _ulti_check(CState* s) noexcept nogil:
    """1 if soloist won the last trick with trump 7, else 0."""
    if s.trump < 0 or s.lt_winner < 0:
        return 0
    cdef int t7 = s.trump * 8  # trump SEVEN (rank 0)
    cdef int i
    for i in range(C_NP):
        if s.lt_c[i] == t7 and s.lt_p[i] == s.soloist:
            return 1 if s.lt_winner == s.soloist else 0
    return 0

# ---------------------------------------------------------------------------
#  Pure-Ulti evaluator (binary: soloist won last trick with trump-7 ⇔ +1)
#
# Modular per-contract solver. The bidding-time evaluator composes V_ulti
# with V_parti and other fundamental contract values; here we only ask
# "did the soloist take trick 10 with the trump-7?". 7esre tartás (already
# in _legal) enforces that the soloist doesn't voluntarily release the 7
# before the last trick.
# ---------------------------------------------------------------------------

cdef int _soloist_ulti_proven_safe(CState* s) noexcept nogil:
    """Úr vagyok: soloist is on lead at a trick boundary and can guarantee
    winning every remaining trick (including the last with trump-7).

    Conditions (all required, all use live in-hand cards):
      (1) tc_n == 0 (trick boundary)
      (2) trump exists and trump-7 is in sol's hand
      (3) current player is sol (sol leads)
      (4) no opponent holds any trump
      (5) for every non-trump suit S where sol holds cards,
          sol's MIN(S) strictly exceeds opp's MAX(S) — "no holes"

    Why MIN>MAX and not MAX>MAX: opp plays optimally on follow. If sol leads
    sol's max in S and opp also has cards in S, opp will sacrifice their
    LOWEST card (which loses anyway) and save their high card to win when
    sol later leads a smaller card in S. So sol_max>opp_max is not enough
    — opp's saved high can later beat sol's remaining cards in S. Requiring
    sol's MIN > opp's MAX removes any path-dependent loss in that suit.

    Soundness: sol leads each remaining trick. Lead any non-trump-7 card.
      * non-trump in S where sol holds: sol's card ≥ sol_min(S) > opp_max(S);
        opp follows with a loser or, if void, discards (no trumps to ruff).
      * non-7 trump lead: opp can't follow trump (no trumps) so discards.
    Sol always wins → sol stays on lead → sol leads trump-7 in trick 10.
    Opp has no trump to over-ruff → sol wins last trick with trump-7 → WIN.

    Opp cards in a suit S where sol has none are stranded (opp never leads).
    """
    if s.tc_n != 0:
        return 0
    if s.trump < 0:
        return 0
    cdef int sol = s.soloist
    if _cur(s) != sol:
        return 0
    cdef int t7 = s.trump * 8
    cdef unsigned int t7_bit = 1u << t7
    if (s.hands[sol] & t7_bit) == 0:
        return 0

    cdef unsigned int sol_hand = s.hands[sol]
    cdef unsigned int opp_hand = 0
    cdef int p, suit, c, str_val
    cdef unsigned int h
    cdef int trump_suit = s.trump

    for p in range(C_NP):
        if p != sol:
            opp_hand |= s.hands[p]

    cdef int sol_min[4]
    cdef int opp_max[4]
    cdef int has_sol[4]
    for suit in range(4):
        sol_min[suit] = 99
        opp_max[suit] = -1
        has_sol[suit] = 0

    h = sol_hand
    while h != 0:
        c = _ctz(h)
        suit = _suit(c)
        if suit != trump_suit:
            str_val = _str_n(c)
            if str_val < sol_min[suit]:
                sol_min[suit] = str_val
            has_sol[suit] = 1
        h &= h - 1

    h = opp_hand
    while h != 0:
        c = _ctz(h)
        suit = _suit(c)
        if suit == trump_suit:
            return 0    # opp holds a trump
        str_val = _str_n(c)
        if str_val > opp_max[suit]:
            opp_max[suit] = str_val
        h &= h - 1

    for suit in range(4):
        if suit == trump_suit:
            continue
        if has_sol[suit] and sol_min[suit] <= opp_max[suit]:
            return 0
    return 1


# Outcome code for early-terminal: 0 = not early, +1 = early WIN, -1 = early LOSS.
cdef int _early_ulti_outcome(CState* s) noexcept nogil:
    if s.trump < 0:
        return -1   # no trump → ulti impossible
    cdef int t7 = s.trump * 8
    cdef unsigned int t7_bit = 1u << t7
    cdef int i

    if (s.hands[0] | s.hands[1] | s.hands[2]) & t7_bit:
        if (s.hands[s.soloist] & t7_bit) == 0:
            return -1   # defender holds trump-7 → cannot win
        # soloist holds trump-7 → check úr predicate
        if _g_ulti_proven_safe and _soloist_ulti_proven_safe(s):
            return 1
        return 0
    # trump-7 already played
    for i in range(s.tc_n):
        if s.tc_c[i] == t7:
            if s.tc_p[i] == s.soloist and s.trick_no == C_TRICKS - 1:
                return 0   # in flight in the last trick → wait for terminal
            return -1
    # not in any hand and not in the current trick → played in a prior trick
    return -1


cdef float _term_ulti(CState* s) noexcept nogil:
    if s.trick_no >= C_TRICKS:
        if _ulti_check(s):
            return C_ULTI_WIN
        return C_ULTI_LOSE
    # Early terminal: ask the outcome helper.
    if _early_ulti_outcome(s) > 0:
        return C_ULTI_WIN
    return C_ULTI_LOSE

cdef int _early_ulti(CState* s) noexcept nogil:
    """Early-terminal iff trump-7's fate is sealed (LOSS) or úr predicate
    fires (WIN). See ``_early_ulti_outcome`` for the full case analysis."""
    return _early_ulti_outcome(s) != 0

cdef void _bounds_ulti(CState* s, float* lo, float* hi) noexcept nogil:
    lo[0] = C_ULTI_LOSE
    hi[0] = C_ULTI_WIN

# ---------------------------------------------------------------------------
#  Multi-payoff evaluator (generic weighted-sum of atomic indicators)
#
# Single C contract that minimaxes a weighted sum of named indicator
# functions. Each component is a pure function of the final position with
# static bounds; the weight vector lives in module globals set from Python
# via ``set_multi_weights``. The oracle layer translates a BidSet into the
# right weight set.
#
# v1 components (4/5/8/9 are stubs — always return 0 until marriages
# are threaded into CState):
#   1. parti_pts          [0, 90]   sol's trick + marriage points
#   2. silent_ulti_signed [-2, 2]   +1 sol won 7, -2 sol bukott,
#                                   -1 def won 7, +2 def bukott
#   3. silent_durchmars   [0, 1]    sol took all 10 tricks
#   4. silent_40          [0, 1]    sol declared trump K+Q (STUB)
#   5. silent_20          [0, 1]    sol declared non-trump K+Q (STUB)
#   6. score_geq_100      [0, 1]    sol's scores >= 100
#   7. sol_tricks_zero    [0, 1]    sol took 0 tricks (betli)
#   8. def_40            [-1, 0]    def declared trump K+Q (STUB)
#   9. def_20            [-1, 0]    def declared non-trump K+Q (STUB)
#
# IMPORTANT: TT must be cleared between solves when weights change.
# The default solve_root(clear_tt=1) handles this automatically; callers
# walking PVs with clear_tt=0 across weight changes will hit stale cache.
# ---------------------------------------------------------------------------

cdef float _g_multi_w_parti_pts          = 0.0
cdef float _g_multi_w_silent_ulti        = 0.0
cdef float _g_multi_w_silent_durchmars   = 0.0
cdef float _g_multi_w_silent_40          = 0.0
cdef float _g_multi_w_silent_20          = 0.0
cdef float _g_multi_w_score_geq_100      = 0.0
cdef float _g_multi_w_sol_tricks_zero    = 0.0
cdef float _g_multi_w_def_40             = 0.0
cdef float _g_multi_w_def_20             = 0.0

# Precomputed static bounds for the weighted sum (excluding parti_pts,
# which is per-state). Recomputed in set_multi_weights.
cdef float _g_multi_static_lo = 0.0
cdef float _g_multi_static_hi = 0.0

# Diagnostic toggle for the EV_MULTI dominance cull (_cull_parti_blocks).
# 1 = on (production default); 0 = swap in _cull_noop. Used to verify the cull
# never changes a multi value (esp. that it never prunes the silent-ulti
# trump-7 line). See set_multi_cull(). Default 1 → no behaviour change.
cdef int _g_multi_cull_enabled = 1


cdef inline void _multi_recompute_bounds() noexcept nogil:
    """Recompute the static contribution to lo/hi from non-parti-pts
    components. Called whenever weights change."""
    global _g_multi_static_lo, _g_multi_static_hi
    cdef float lo = 0.0
    cdef float hi = 0.0
    cdef float w
    # silent_ulti_signed: [-2, 2]
    w = _g_multi_w_silent_ulti
    if w >= 0:
        lo += w * -2.0; hi += w * 2.0
    else:
        lo += w * 2.0;  hi += w * -2.0
    # silent_durchmars: [0, 1]
    w = _g_multi_w_silent_durchmars
    if w >= 0:
        hi += w
    else:
        lo += w
    # silent_40: [0, 1]  (stub, always 0 — bound contributes 0)
    w = _g_multi_w_silent_40
    if w >= 0:
        hi += w
    else:
        lo += w
    # silent_20: [0, 1]  (stub)
    w = _g_multi_w_silent_20
    if w >= 0:
        hi += w
    else:
        lo += w
    # score_geq_100: [0, 1]
    w = _g_multi_w_score_geq_100
    if w >= 0:
        hi += w
    else:
        lo += w
    # sol_tricks_zero: [0, 1]
    w = _g_multi_w_sol_tricks_zero
    if w >= 0:
        hi += w
    else:
        lo += w
    # def_40: [-1, 0]  (stub)
    w = _g_multi_w_def_40
    if w >= 0:
        lo += w * -1.0
    else:
        hi += w * -1.0
    # def_20: [-1, 0]  (stub)
    w = _g_multi_w_def_20
    if w >= 0:
        lo += w * -1.0
    else:
        hi += w * -1.0
    _g_multi_static_lo = lo
    _g_multi_static_hi = hi


def set_multi_weights(
    float parti_pts=0.0,
    float silent_ulti=0.0,
    float silent_durchmars=0.0,
    float silent_40=0.0,
    float silent_20=0.0,
    float score_geq_100=0.0,
    float sol_tricks_zero=0.0,
    float def_40=0.0,
    float def_20=0.0,
):
    """Set the EV_MULTI weight vector. Caller MUST clear the TT after
    changing weights (solve_root(..., clear_tt=1) does this by default)."""
    global _g_multi_w_parti_pts, _g_multi_w_silent_ulti
    global _g_multi_w_silent_durchmars, _g_multi_w_silent_40, _g_multi_w_silent_20
    global _g_multi_w_score_geq_100, _g_multi_w_sol_tricks_zero
    global _g_multi_w_def_40, _g_multi_w_def_20
    _g_multi_w_parti_pts        = parti_pts
    _g_multi_w_silent_ulti      = silent_ulti
    _g_multi_w_silent_durchmars = silent_durchmars
    _g_multi_w_silent_40        = silent_40
    _g_multi_w_silent_20        = silent_20
    _g_multi_w_score_geq_100    = score_geq_100
    _g_multi_w_sol_tricks_zero  = sol_tricks_zero
    _g_multi_w_def_40           = def_40
    _g_multi_w_def_20           = def_20
    _multi_recompute_bounds()


def get_multi_weights():
    return dict(
        parti_pts        = _g_multi_w_parti_pts,
        silent_ulti      = _g_multi_w_silent_ulti,
        silent_durchmars = _g_multi_w_silent_durchmars,
        silent_40        = _g_multi_w_silent_40,
        silent_20        = _g_multi_w_silent_20,
        score_geq_100    = _g_multi_w_score_geq_100,
        sol_tricks_zero  = _g_multi_w_sol_tricks_zero,
        def_40           = _g_multi_w_def_40,
        def_20           = _g_multi_w_def_20,
    )


def set_multi_cull(int enabled):
    """Diagnostic toggle for the EV_MULTI dominance cull.
    1 = _cull_parti_blocks (production default); 0 = _cull_noop (no pruning).
    Used to verify the cull never changes a multi value — in particular that it
    never prunes the silent-ulti trump-7 line for the soloist OR a defender.
    Leave at 1 in production."""
    global _g_multi_cull_enabled
    _g_multi_cull_enabled = 1 if enabled else 0


def get_multi_cull():
    return _g_multi_cull_enabled


cdef inline float _multi_silent_ulti_signed(CState* s) noexcept nogil:
    """Asymmetric bukott rule: +1 sol won 7, -2 sol bukott, -1 def won 7,
    +2 def bukott (someone other than the def who played 7 took the trick),
    0 if no trump-7 played in last trick."""
    if s.trump < 0 or s.lt_winner < 0:
        return 0.0
    cdef int t7 = s.trump * 8   # trump SEVEN
    cdef int i
    cdef int player_who_played = -1
    for i in range(C_NP):
        if s.lt_c[i] == t7:
            player_who_played = s.lt_p[i]
            break
    if player_who_played < 0:
        return 0.0
    if player_who_played == s.soloist:
        return 1.0 if s.lt_winner == s.soloist else -2.0
    # defender played the 7
    return -1.0 if s.lt_winner == player_who_played else 2.0


cdef float _term_multi(CState* s) noexcept nogil:
    """Weighted-sum terminal value across all 9 components."""
    cdef float v = 0.0
    cdef float parti_val
    if _g_multi_w_parti_pts != 0.0:
        parti_val = 1.0 if s.scores[s.soloist] > _def_score(s) else -1.0
        v += _g_multi_w_parti_pts * parti_val
    if _g_multi_w_silent_ulti != 0.0:
        v += _g_multi_w_silent_ulti * _multi_silent_ulti_signed(s)
    if _g_multi_w_silent_durchmars != 0.0:
        if s.tricks_won[s.soloist] == C_TRICKS:
            v += _g_multi_w_silent_durchmars
    # silent_40, silent_20, def_40, def_20: stubs (no marriage state in CState)
    if _g_multi_w_score_geq_100 != 0.0:
        if s.scores[s.soloist] >= 100:
            v += _g_multi_w_score_geq_100
    if _g_multi_w_sol_tricks_zero != 0.0:
        if s.tricks_won[s.soloist] == 0:
            v += _g_multi_w_sol_tricks_zero
    return v


cdef int _early_multi(CState* s) noexcept nogil:
    # v1: no early termination. The alpha-beta engine still terminates at
    # trick 10 via s.trick_no >= C_TRICKS. Component-specific early shortcuts
    # (sol_tricks_zero=1 + sol took a trick → early LOSE; silent_durchmars=1
    # + sol lost a trick → early LOSE) are deferred to v2.
    return 0


cdef void _bounds_multi(CState* s, float* lo, float* hi) noexcept nogil:
    """Bounds = static contribution (precomputed) + per-state parti_pts.
    Parti channel is now binary win/loss, so its contribution collapses
    to ±w once the outcome is forced and is bounded by ±|w| otherwise."""
    cdef float bl = _g_multi_static_lo
    cdef float bh = _g_multi_static_hi
    cdef float w = _g_multi_w_parti_pts
    cdef int sp, dp, rem
    cdef float plo, phi
    if w != 0.0:
        sp = s.scores[s.soloist]
        dp = _def_score(s)
        rem = _rem_pts(s)
        if sp > dp + rem:
            plo = 1.0; phi = 1.0
        elif sp + rem <= dp:
            plo = -1.0; phi = -1.0
        else:
            plo = -1.0; phi = 1.0
        if w >= 0:
            bl += w * plo
            bh += w * phi
        else:
            bl += w * phi
            bh += w * plo
    lo[0] = bl
    hi[0] = bh

# ---------------------------------------------------------------------------
#  Evaluator factory
# ---------------------------------------------------------------------------

cdef ContractEval _get_eval(int contract_id) noexcept nogil:
    cdef ContractEval ev
    ev.cull     = _cull_noop
    ev.use_tt   = 1
    ev.tt_extra = _tt_extra_full   # parti-safe default; binary contracts may override
    if contract_id == EV_BETLI:
        ev.terminal   = _term_betli
        ev.early_term = _early_betli
        ev.bounds     = _bounds_betli
        ev.order      = _order_betli
        ev.cull       = _cull_betli_dominant
    elif contract_id == EV_DURCHMARS:
        ev.terminal   = _term_dm
        ev.early_term = _early_dm
        ev.bounds     = _bounds_dm
        ev.order      = _order_default
        ev.cull       = _cull_parti_blocks   # binary outcome, no point splits
        ev.tt_extra   = _tt_extra_slim       # binary contract: drop scores+tricks_won
    elif contract_id == EV_ULTI:
        ev.terminal   = _term_ulti
        ev.early_term = _early_ulti
        ev.bounds     = _bounds_ulti
        if _g_ulti_order_id == 1:
            ev.order  = _order_ulti_no_pts
        elif _g_ulti_order_id == 2:
            ev.order  = _order_ulti_pull_duck
        elif _g_ulti_order_id == 3:
            ev.order  = _order_ulti_t7_first
        elif _g_ulti_order_id == 4:
            ev.order  = _order_ulti_human
        else:
            ev.order  = _order_default
        # _term_ulti reads s.lt_* only at the terminal (trick_no >= 10),
        # which is checked before the TT probe in _ab. So caching the
        # subtree value at interior nodes is path-independent.
        ev.cull       = _cull_parti_blocks   # trump-7 isolated via has_ulti guard
        ev.use_tt     = 1
        if _g_ulti_tt_slim:
            ev.tt_extra = _tt_extra_slim    # binary outcome: scores/tricks_won don't affect cached value
    elif contract_id == EV_MULTI:
        ev.terminal   = _term_multi
        ev.early_term = _early_multi
        ev.bounds     = _bounds_multi
        ev.order      = _order_default
        # Colored-contract recipe (parti_pts active, point values matter):
        # _cull_parti_blocks is the right cull. It auto-isolates the
        # trump-7 when silent_ulti weight is non-zero (see the gate in
        # _cull_parti_blocks itself), for whoever holds it (sol OR def), so
        # silent ulti is never pruned. _g_multi_cull_enabled (default 1) can
        # swap in _cull_noop to verify that equivalence — see set_multi_cull().
        ev.cull       = _cull_parti_blocks if _g_multi_cull_enabled else _cull_noop
        ev.use_tt     = 1
        ev.tt_extra   = _tt_extra_full      # parti-safe (covers parti_pts contribution)
    else:  # EV_PARTI (default)
        ev.terminal   = _term_parti
        ev.early_term = _early_parti
        ev.bounds     = _bounds_parti
        if _g_parti_order_id == 1:
            ev.order  = _order_parti_pts_first
        elif _g_parti_order_id == 2:
            ev.order  = _order_parti_winner_strength
        else:
            ev.order  = _order_default
        ev.cull       = _cull_parti_blocks
    return ev

# ===========================================================================
#  Transposition table (open-addressing, single-probe replacement)
# ===========================================================================
# Fixed-size global table. Cleared at the start of each solve_root /
# solve_best. Keys include the three hand bitmasks plus a packed
# ``extra`` field holding all other state that affects the value
# (current-trick partial, leader, scores[soloist], tricks_won[soloist]).
# The pure-ulti contract reads s.lt_* only at the terminal, which is
# checked before the TT probe, so TT cutoffs are safe there too.

# TT sizing chosen by empirical sweep on the PIMC32 workload (32 ulti-
# biased opening solves, multi[parti+ulti] solver). Larger TT is NOT
# strictly better: it has to fit each CPU core's L2 cache to be worth
# the memory bandwidth. Sweep shows a clean U-curve with floor at
# tt_log2=17..18 (4-8 MB / TT, fits in M-series L2 of 4-12 MB/perf-core).
# Going smaller pays collision-rate cost (node count explodes); going
# bigger pays cache-miss cost.
#
# Single-batch median wall (32-position PIMC32 batch):
#   tt_log2  mem/TT   1-thread  8-thread  node-count
#       16    2 MB     724 ms    237 ms      17 M
#       17    4 MB     600 ms    195 ms      14 M
#   *   18    8 MB     528 ms    193 ms      12 M    ← chosen
#       20   32 MB     626 ms    219 ms      10 M
#       22  128 MB     763 ms    254 ms      10 M
#       24  512 MB    1069 ms    461 ms      10 M    ← old default
DEF TT_LOG2 = 18            # 1 << 18 = 262,144 entries × 32 B = 8 MB
DEF TT_SIZE = 262144
DEF TT_MASK = 262143

DEF TT_EMPTY = 0
DEF TT_EXACT = 1
DEF TT_LOWER = 2
DEF TT_UPPER = 3

cdef struct TTEntry:
    unsigned int        h0          # 4
    unsigned int        h1          # 4
    unsigned int        h2          # 4
    unsigned long long  extra       # 8 (after 4B padding to align)
    float               value       # 4
    unsigned char       bound       # 1 (TT_EMPTY=0, TT_EXACT=1, TT_LOWER=2, TT_UPPER=3)
    signed char         best_move   # 1 (card id 0..31; -1 = no hint)
    # 2 bytes tail padding → 32 B total (same as before adding best_move).

cdef TTEntry _tt_arr[TT_SIZE]      # ~32 MB (singleton context's TT buffer)


# Per-call mutable search state. The singleton ``_g_ctx`` points at the
# static ``_tt_arr`` and backs every legacy entry point (solve_root /
# solve_best / principal_variation / pis.solve_all). A future
# solve_all_batch will allocate per-thread contexts with their own
# malloc'd TT buffers and call into the same _ab path.
cdef struct SearchContext:
    TTEntry*     tt              # pointer to the TT buffer
    unsigned int tt_size         # number of entries in tt (power of 2)
    unsigned int tt_mask         # tt_size - 1
    long long    nodes
    long long    cuts
    long long    tt_hits
    long long    tt_stores
    long long    terminals
    long long    bounds_cuts
    long long    no_moves


cdef SearchContext _g_ctx     # initialized in _init_g_ctx() at module load


cdef void _init_g_ctx() noexcept nogil:
    _g_ctx.tt          = _tt_arr
    _g_ctx.tt_size     = TT_SIZE
    _g_ctx.tt_mask     = TT_MASK
    _g_ctx.nodes       = 0
    _g_ctx.cuts        = 0
    _g_ctx.tt_hits     = 0
    _g_ctx.tt_stores   = 0
    _g_ctx.terminals   = 0
    _g_ctx.bounds_cuts = 0
    _g_ctx.no_moves    = 0


cdef inline void _ctx_reset_counters(SearchContext* ctx) noexcept nogil:
    ctx.nodes       = 0
    ctx.cuts        = 0
    ctx.tt_hits     = 0
    ctx.tt_stores   = 0
    ctx.terminals   = 0
    ctx.bounds_cuts = 0
    ctx.no_moves    = 0


cdef inline void _tt_clear(SearchContext* ctx) noexcept nogil:
    cdef unsigned int i
    for i in range(ctx.tt_size):
        ctx.tt[i].bound = TT_EMPTY


cdef inline unsigned long long _tt_extra_full(CState* s) noexcept nogil:
    """Full TT key — includes scores+tricks_won. Required for parti, where
    the cached value is the *absolute* soloist score; paths reaching the
    same hand state with different prior captures have genuinely different
    cached values.

    Unused in-flight trick slots (tc_p[1]/tc_c[1] when tc_n<2, both when
    tc_n=0) are gated by tc_n. Otherwise they carry stale residue from
    the just-completed trick (``_apply`` doesn't clear them on completion),
    fragmenting every cache entry by its arrival path."""
    cdef unsigned long long e = 0
    e |= (<unsigned long long>(s.leader  & 0x3))       << 0
    e |= (<unsigned long long>(s.trick_no & 0xF))      << 2
    e |= (<unsigned long long>(s.tc_n    & 0x3))       << 6
    if s.tc_n >= 1:
        e |= (<unsigned long long>(s.tc_p[0] & 0x3))   << 8
        e |= (<unsigned long long>(s.tc_c[0] & 0x1F))  << 12
    if s.tc_n >= 2:
        e |= (<unsigned long long>(s.tc_p[1] & 0x3))   << 10
        e |= (<unsigned long long>(s.tc_c[1] & 0x1F))  << 17
    e |= (<unsigned long long>(s.scores[s.soloist] & 0xFF)) << 22
    e |= (<unsigned long long>(s.tricks_won[s.soloist] & 0xF)) << 30
    return e

cdef inline unsigned long long _tt_extra_slim(CState* s) noexcept nogil:
    """Slim TT key — drops scores+tricks_won. Safe for binary contracts
    whose terminal value depends only on the remaining cards / who plays
    the decisive card (ulti: trump-7 in trick 10). Past captured scores
    are dead weight in the key for such contracts.

    Same tc_n gating as ``_tt_extra_full`` — see its docstring."""
    cdef unsigned long long e = 0
    e |= (<unsigned long long>(s.leader  & 0x3))       << 0
    e |= (<unsigned long long>(s.trick_no & 0xF))      << 2
    e |= (<unsigned long long>(s.tc_n    & 0x3))       << 6
    if s.tc_n >= 1:
        e |= (<unsigned long long>(s.tc_p[0] & 0x3))   << 8
        e |= (<unsigned long long>(s.tc_c[0] & 0x1F))  << 12
    if s.tc_n >= 2:
        e |= (<unsigned long long>(s.tc_p[1] & 0x3))   << 10
        e |= (<unsigned long long>(s.tc_c[1] & 0x1F))  << 17
    return e


# Toggle: 1 = slim TT key for EV_ULTI (default), 0 = full (parti-style, A/B baseline).
cdef int _g_ulti_tt_slim = 1

def set_ulti_tt_slim(int enabled):
    """Toggle the slim TT key for EV_ULTI. 1=slim (drop scores+tricks_won),
    0=full (parti-style key, for A/B benchmarking)."""
    global _g_ulti_tt_slim
    _g_ulti_tt_slim = 1 if enabled else 0

def get_ulti_tt_slim():
    return _g_ulti_tt_slim


cdef inline unsigned long long _splitmix64(unsigned long long x) noexcept nogil:
    """SplitMix64 avalanche: each input bit affects roughly half the output
    bits. Standard finalizer for low-entropy keys; better spread than a
    single multiply when the input has few set bits."""
    x = (x ^ (x >> 30)) * <unsigned long long>0xBF58476D1CE4E5B5
    x = (x ^ (x >> 27)) * <unsigned long long>0x94D049BB133111EB
    x = x ^ (x >> 31)
    return x

cdef inline unsigned int _tt_idx(unsigned int h0, unsigned int h1,
                                  unsigned int h2,
                                  unsigned long long extra) noexcept nogil:
    """Hash four key components into a 32-bit slot id. Callers mask with
    their own TT's mask: ``idx = _tt_idx(...) & ctx.tt_mask``."""
    cdef unsigned long long k
    k = _splitmix64(<unsigned long long>h0)
    k ^= _splitmix64(<unsigned long long>h1 + <unsigned long long>0x9E3779B97F4A7C15)
    k ^= _splitmix64(<unsigned long long>h2 + <unsigned long long>0xBB67AE8584CAA73B)
    k ^= _splitmix64(extra              + <unsigned long long>0x3C6EF372FE94F82A)
    k = _splitmix64(k)
    return <unsigned int>k


# ===========================================================================
#  Alpha-beta minimax (contract-agnostic)
# ===========================================================================

cdef float _ab(CState* s, float alpha, float beta,
               ContractEval* ev, SearchContext* ctx) noexcept nogil:
    """Recursive alpha-beta from soloist's perspective.

    ``ev`` controls terminal scoring, early termination, bounds pruning,
    move ordering, dominance culling, and whether to use the TT. ``ctx``
    carries the per-call mutable state (TT buffer + diagnostic counters)
    so we can run multiple searches in parallel by giving each thread
    its own context. The search engine itself stays contract-agnostic.
    """
    ctx.nodes += 1

    # Terminal or early-terminal
    if s.trick_no >= C_TRICKS or ev.early_term(s):
        ctx.terminals += 1
        return ev.terminal(s)

    # Bounds (futility) pruning
    cdef float lo, hi
    ev.bounds(s, &lo, &hi)
    if hi <= alpha:
        ctx.bounds_cuts += 1
        return hi
    if lo >= beta:
        ctx.bounds_cuts += 1
        return lo

    cdef float alpha0 = alpha
    cdef float beta0  = beta

    # ── Transposition-table probe ──────────────────────────────────────────
    cdef unsigned long long extra = 0
    cdef unsigned int idx = 0
    cdef TTEntry* e = NULL
    cdef int tt_best = -1     # best-move hint from a prior visit to this state
    if ev.use_tt:
        extra = ev.tt_extra(s)
        idx   = _tt_idx(s.hands[0], s.hands[1], s.hands[2], extra) & ctx.tt_mask
        e     = &ctx.tt[idx]
        if (e.bound != TT_EMPTY
            and e.h0 == s.hands[0]
            and e.h1 == s.hands[1]
            and e.h2 == s.hands[2]
            and e.extra == extra):
            # We have a TT entry for this exact state. Two uses:
            #   (1) if bound is tight enough vs current alpha/beta, return.
            #   (2) otherwise, capture best_move as an ordering hint.
            tt_best = e.best_move
            if e.bound == TT_EXACT:
                ctx.tt_hits += 1
                return e.value
            elif e.bound == TT_LOWER:
                if e.value >= beta:
                    ctx.tt_hits += 1
                    return e.value
                if e.value > alpha:
                    alpha = e.value
            else:   # TT_UPPER
                if e.value <= alpha:
                    ctx.tt_hits += 1
                    return e.value
                if e.value < beta:
                    beta = e.value
            if alpha >= beta:
                ctx.tt_hits += 1
                return e.value

    cdef int player = _cur(s)
    cdef int maxi = (player == s.soloist)

    cdef Moves mv
    _legal(s, &mv)
    ev.cull(&mv, s, maxi)
    ev.order(&mv, s, maxi)

    if mv.n == 0:
        ctx.no_moves += 1
        return ev.terminal(s)

    # Best-move ordering: if TT had a best-move hint for this state and
    # it survived the cull, swap it to position 0. Massively improves
    # alpha-beta cutoffs since the best move from the prior visit is
    # very likely to also be the best (or near-best) move now.
    cdef int j
    cdef int tmp
    if tt_best >= 0:
        for j in range(mv.n):
            if mv.c[j] == tt_best:
                if j != 0:
                    tmp = mv.c[0]
                    mv.c[0] = mv.c[j]
                    mv.c[j] = tmp
                break

    # ── Children loop with PVS (principal variation search) ──
    # First child: search with the full [alpha, beta] window.
    # Subsequent children: try a null window [alpha, alpha+eps] (max) /
    # [beta-eps, beta] (min) — if our move-ordering is good (it is, via
    # _cull_parti_blocks), most non-first searches will fail low and the
    # null window proves "no improvement" cheaply. Only if the null search
    # fails high do we re-search with the real window.
    #
    # PVS_EPSILON = 1.0: all our terminal scores are integer-valued
    # (parti_pts in 0..90; silent_ulti_signed × weight 2 → ±4 multiples
    # of 2; binary contracts return 0/10). A window of width 1 never
    # splits two reachable values. Smaller would risk float-precision
    # false-negatives; larger would miss cuts.
    cdef float PVS_EPSILON = 1.0
    cdef float val, v
    cdef int i
    cdef int best_i = 0
    cdef Undo u

    if maxi:
        val = -C_INF
        for i in range(mv.n):
            _apply(s, mv.c[i], &u)
            if i == 0:
                v = _ab(s, alpha, beta, ev, ctx)
            else:
                # Null-window probe. If the result fails high (> alpha),
                # re-search with the real window starting from v as the
                # known lower bound — tighter than restarting at alpha.
                v = _ab(s, alpha, alpha + PVS_EPSILON, ev, ctx)
                if v > alpha and v < beta:
                    v = _ab(s, v, beta, ev, ctx)
            _undo(s, &u)
            if v > val:
                val = v
                best_i = i
            if val > alpha:
                alpha = val
            if alpha >= beta:
                ctx.cuts += 1
                break
    else:
        val = C_INF
        for i in range(mv.n):
            _apply(s, mv.c[i], &u)
            if i == 0:
                v = _ab(s, alpha, beta, ev, ctx)
            else:
                # Null-window probe from above for the minimiser.
                v = _ab(s, beta - PVS_EPSILON, beta, ev, ctx)
                if v < beta and v > alpha:
                    v = _ab(s, alpha, v, ev, ctx)
            _undo(s, &u)
            if v < val:
                val = v
                best_i = i
            if val < beta:
                beta = val
            if alpha >= beta:
                ctx.cuts += 1
                break

    # ── Transposition-table store ──────────────────────────────────────────
    if ev.use_tt:
        e = &ctx.tt[idx]
        if val <= alpha0:
            e.bound = TT_UPPER
        elif val >= beta0:
            e.bound = TT_LOWER
        else:
            e.bound = TT_EXACT
        e.best_move = mv.c[best_i]
        e.value = val
        e.h0    = s.hands[0]
        e.h1    = s.hands[1]
        e.h2    = s.hands[2]
        e.extra = extra
        ctx.tt_stores += 1

    return val

# ===========================================================================
#  Python ↔ C conversion
# ===========================================================================

_SUIT_MAP = {}
_SUIT_RMAP = {}
_CARD_CACHE = {}

def _init_maps():
    from ultisolver.games.ulti.cards import Suit, Rank, Card, ALL_SUITS, ALL_RANKS
    for i, s in enumerate(ALL_SUITS):
        _SUIT_MAP[s] = i
        _SUIT_RMAP[i] = s
    for s in ALL_SUITS:
        for r in ALL_RANKS:
            cid = _SUIT_MAP[s] * 8 + int(r)
            _CARD_CACHE[cid] = Card(s, r)

_init_maps()
_init_g_ctx()

# Contract string → C enum
_CONTRACT_MAP = {
    "parti": EV_PARTI,
    "betli": EV_BETLI,
    "durchmars": EV_DURCHMARS,
    "ulti": EV_ULTI,
    "multi": EV_MULTI,
}


cdef inline int _c2id(object card):
    return _SUIT_MAP[card.suit] * 8 + int(card.rank)

cdef inline object _id2c(int card_id):
    return _CARD_CACHE[card_id]


cdef CState _to_cs(object gs):
    """Convert Python GameState → CState."""
    cdef CState s
    cdef int i

    # Solver invariant: per-hand size <= C_MAX_MOVES (=10). _legal,
    # _order_*, _cull_* write into fixed-size stack buffers of that
    # length; a single oversized hand corrupts the stack and SIGABRTs.
    # Bounce here before the C path can dereference the bad input.
    for i in range(C_NP):
        if len(gs.hands[i]) > C_MAX_MOVES:
            raise ValueError(
                f"GameState.hands[{i}] has {len(gs.hands[i])} cards; "
                f"solver max is {C_MAX_MOVES}"
            )

    for i in range(C_NP):
        s.hands[i] = 0
        s.scores[i] = gs.scores[i]
        s.tricks_won[i] = len(gs.captured[i]) // C_NP

    for i in range(C_NP):
        for card in gs.hands[i]:
            s.hands[i] |= _bit(_c2id(card))

    s.trump = _SUIT_MAP[gs.trump] if gs.trump is not None else C_NO_TRUMP
    s.betli = 1 if gs.betli else 0
    s.soloist = gs.soloist
    s.leader = gs.leader
    s.trick_no = gs.trick_no
    s.has_ulti = 1 if gs.has_ulti else 0
    s.binary = 0       # overridden in _setup once ev_id is known

    s.tc_n = len(gs.trick_cards)
    for i in range(s.tc_n):
        p, c = gs.trick_cards[i]
        s.tc_p[i] = p
        s.tc_c[i] = _c2id(c)

    # Last-trick info
    if gs.last_trick is not None:
        s.lt_winner = gs.last_trick.winner
        for i in range(C_NP):
            s.lt_p[i] = gs.last_trick.players[i]
            s.lt_c[i] = _c2id(gs.last_trick.cards[i])
    else:
        s.lt_winner = -1
        for i in range(C_NP):
            s.lt_p[i] = -1
            s.lt_c[i] = -1

    return s


def _detect_contract(gs):
    """Auto-detect contract type from GameState flags."""
    if getattr(gs, 'training_mode', None) == "durchmars":
        return "durchmars"
    if gs.betli:
        return "betli"
    if getattr(gs, 'has_ulti', False):
        return "ulti"
    return "parti"

# ===========================================================================
#  Public API
# ===========================================================================

# List of available contracts for discovery
CONTRACTS = list(_CONTRACT_MAP.keys())


# ---------------------------------------------------------------------------
#  Shared setup + inner search helpers
# ---------------------------------------------------------------------------

cdef CState _setup(gs, contract, int clear_tt, ContractEval* out_ev):
    """Common prologue: reset singleton-context counters, resolve contract,
    optionally clear singleton TT, convert ``gs`` to the C state. Used by
    every public legacy entry point (solve_root / solve_best / etc.)."""
    _ctx_reset_counters(&_g_ctx)

    if contract is None:
        contract = _detect_contract(gs)
    cdef int ev_id = _CONTRACT_MAP.get(contract, EV_PARTI)
    out_ev[0] = _get_eval(ev_id)
    if clear_tt and out_ev.use_tt:
        _tt_clear(&_g_ctx)

    cdef CState s = _to_cs(gs)
    # binary=1 affects culls (binary contracts ignore points). EV_MULTI is
    # parti-shaped by default (parti_pts is continuous); stay scalar.
    s.binary = 1 if (ev_id == EV_DURCHMARS or ev_id == EV_ULTI) else 0
    return s


cdef int _solve_best_inner(CState* s, ContractEval* ev,
                            Moves* mv, int* out_best_i,
                            float* out_best_val,
                            SearchContext* ctx) noexcept nogil:
    """Compute the best move at ``s`` and write the legal move list into
    ``mv``. Fills ``out_best_i`` and ``out_best_val`` with the index into
    ``mv.c[]`` and the soloist-perspective minimax value. Returns 0 when
    there are no legal moves. ``ctx`` is threaded through to ``_ab``."""
    cdef int player = _cur(s)
    cdef int maxi   = (player == s.soloist)

    _legal(s, mv)
    ev.order(mv, s, maxi)

    if mv.n == 0:
        return 0

    cdef Undo u
    cdef int i
    cdef int best_i = 0
    cdef float best_val, v

    if mv.n == 1:
        _apply(s, mv.c[0], &u)
        best_val = _ab(s, -C_INF, C_INF, ev, ctx)
        _undo(s, &u)
        out_best_i[0]   = 0
        out_best_val[0] = best_val
        return 1

    if maxi:
        best_val = -C_INF
        for i in range(mv.n):
            _apply(s, mv.c[i], &u)
            v = _ab(s, best_val, C_INF, ev, ctx)
            _undo(s, &u)
            if v > best_val:
                best_val = v; best_i = i
    else:
        best_val = C_INF
        for i in range(mv.n):
            _apply(s, mv.c[i], &u)
            v = _ab(s, -C_INF, best_val, ev, ctx)
            _undo(s, &u)
            if v < best_val:
                best_val = v; best_i = i

    out_best_i[0]   = best_i
    out_best_val[0] = best_val
    return 1


# ---------------------------------------------------------------------------
#  Public solve API
# ---------------------------------------------------------------------------

def solve_root(gs, int max_exact_tricks=C_TRICKS, contract=None,
                int clear_tt=1):
    """Compute exact value for every legal move at the current ply.

    Parameters
    ----------
    gs : GameState
    contract : str or None
        Contract type: "parti", "betli", "durchmars", "ulti".
        If None, auto-detected from ``gs.betli`` / ``gs.has_ulti``.
    clear_tt : int (bool)
        Reset the transposition table before solving. Default ``1``.
        Pass ``0`` to keep the TT warm across consecutive solves on
        related positions (e.g. when walking a principal variation).
    max_exact_tricks : int
        Kept for API compatibility; ignored.

    Returns
    -------
    dict[Card, float]
        Value for each legal move from the soloist's perspective.
    """
    cdef ContractEval ev
    cdef CState s = _setup(gs, contract, clear_tt, &ev)
    cdef int player = _cur(&s)
    cdef int maxi = (player == s.soloist)

    cdef Moves mv
    _legal(&s, &mv)
    ev.order(&mv, &s, maxi)

    cdef dict values = {}
    cdef float v
    cdef Undo u
    cdef int i

    for i in range(mv.n):
        _apply(&s, mv.c[i], &u)
        v = _ab(&s, -C_INF, C_INF, &ev, &_g_ctx)
        _undo(&s, &u)
        values[_id2c(mv.c[i])] = v

    return values


def solve_best(gs, int max_exact_tricks=C_TRICKS, contract=None,
                int clear_tt=1):
    """Find the best move and its exact minimax value.

    Parameters mirror ``solve_root``. Pass ``clear_tt=0`` to reuse the TT
    warmed by a previous solve on a related position — that's how
    ``principal_variation`` walks the game cheaply ply by ply.

    Returns
    -------
    (Card | None, float)
    """
    cdef ContractEval ev
    cdef CState s = _setup(gs, contract, clear_tt, &ev)

    cdef Moves mv
    cdef int best_i
    cdef float best_val
    if not _solve_best_inner(&s, &ev, &mv, &best_i, &best_val, &_g_ctx):
        return None, 0.0
    return _id2c(mv.c[best_i]), best_val


def principal_variation(gs, contract=None):
    """Optimal play sequence from the current position to game end.

    Returns a list of ``(player_id, Card)`` tuples — perfect play from
    here until the game terminates. Walks the game ply by ply via
    repeated inner solves; the TT stays warm across plies so each step
    is mostly TT hits.
    """
    cdef ContractEval ev
    cdef CState s = _setup(gs, contract, 1, &ev)

    cdef Moves mv
    cdef int best_i
    cdef float best_val
    cdef Undo u
    cdef int player

    pv = []
    while s.trick_no < C_TRICKS and not ev.early_term(&s):
        player = _cur(&s)
        if not _solve_best_inner(&s, &ev, &mv, &best_i, &best_val, &_g_ctx):
            break
        pv.append((player, _id2c(mv.c[best_i])))
        _apply(&s, mv.c[best_i], &u)
    return pv


# ===========================================================================
#  Per-thread context API (for solve_all_batch)
# ===========================================================================
# Allocates a SearchContext with its own malloc'd TT buffer so multiple
# threads can run independent searches in parallel without sharing state.
# Caller owns the lifetime — call free_context(handle) when done.

def alloc_context(int tt_log2=18):
    """Allocate a new SearchContext with its own TT buffer.

    ``tt_log2`` sets the TT size: 1 << tt_log2 entries × 32 B/entry.
    Defaults to 18 (262K entries, 8 MB) — matches the singleton's TT
    and is the empirically-measured optimum for PIMC32 workloads
    (see the TT_LOG2 sizing comment above for the sweep table).
    8 MB fits in M-series perf-core L2 cache; larger TTs pay a
    cache-miss cost without buying much in hit rate.

    Returns an opaque integer handle to be passed to solve_all_ctx() /
    get_context_stats() / free_context(). Each thread should own one
    context; multiple threads sharing one context would race on the TT.
    """
    if tt_log2 < 10 or tt_log2 > 31:
        raise ValueError(f"tt_log2={tt_log2} out of range (10..31)")
    cdef unsigned int sz = 1u << tt_log2
    cdef SearchContext* ctx = <SearchContext*>malloc(sizeof(SearchContext))
    if ctx == NULL:
        raise MemoryError("alloc SearchContext")
    # calloc zeroes → all TT_EMPTY (bound=0) and counters=0 implicitly.
    ctx.tt = <TTEntry*>calloc(sz, sizeof(TTEntry))
    if ctx.tt == NULL:
        free(ctx)
        raise MemoryError(f"alloc TT buffer ({sz} entries)")
    ctx.tt_size = sz
    ctx.tt_mask = sz - 1
    ctx.nodes = 0
    ctx.cuts = 0
    ctx.tt_hits = 0
    ctx.tt_stores = 0
    ctx.terminals = 0
    ctx.bounds_cuts = 0
    ctx.no_moves = 0
    return <unsigned long long><void*>ctx


def free_context(unsigned long long handle):
    """Free a context previously returned by alloc_context()."""
    cdef SearchContext* ctx = <SearchContext*><void*><unsigned long long>handle
    if ctx == NULL:
        return
    if ctx.tt != NULL:
        free(ctx.tt)
    free(ctx)


def get_context_stats(unsigned long long handle):
    """Diagnostics from the most recent solve on this context."""
    cdef SearchContext* ctx = <SearchContext*><void*><unsigned long long>handle
    cdef long long n  = ctx.nodes
    cdef long long c  = ctx.cuts
    cdef long long th = ctx.tt_hits
    cdef long long ts = ctx.tt_stores
    cdef long long t  = ctx.terminals
    cdef long long b  = ctx.bounds_cuts
    cdef long long nm = ctx.no_moves
    interior = n - t - b - th - nm
    return {
        "nodes_explored": n,
        "cutoffs": c,
        "pruning_ratio": (<double>c / <double>n) if n > 0 else 0.0,
        "tt_hits":   th,
        "tt_stores": ts,
        "terminals":   t,
        "bounds_cuts": b,
        "no_moves":    nm,
        "interior":    interior,
    }


def solve_all_ctx(gs, contract, unsigned long long ctx_handle,
                   int clear_tt=1):
    """Like solve_all (the legacy solve_root), but uses a caller-provided
    context instead of the singleton. The GIL is released around the
    alpha-beta loop so multiple threads with their own contexts run
    truly in parallel.

    Parameters mirror solve_root. ``clear_tt=1`` clears just *this*
    context's TT (other contexts are untouched).
    """
    cdef SearchContext* ctx = <SearchContext*><void*><unsigned long long>ctx_handle
    cdef ContractEval ev
    _ctx_reset_counters(ctx)

    if contract is None:
        contract = _detect_contract(gs)
    cdef int ev_id = _CONTRACT_MAP.get(contract, EV_PARTI)
    ev = _get_eval(ev_id)
    if clear_tt and ev.use_tt:
        _tt_clear(ctx)

    cdef CState s = _to_cs(gs)
    s.binary = 1 if (ev_id == EV_DURCHMARS or ev_id == EV_ULTI) else 0

    cdef int player = _cur(&s)
    cdef int maxi = (player == s.soloist)

    cdef Moves mv
    _legal(&s, &mv)
    ev.order(&mv, &s, maxi)

    cdef int n_moves = mv.n
    cdef float results[C_MAX_MOVES]
    cdef Undo u
    cdef int i

    # GIL released for the full search loop — this is what gives us
    # threaded parallelism when called by multiple Python threads.
    with nogil:
        for i in range(n_moves):
            _apply(&s, mv.c[i], &u)
            results[i] = _ab(&s, -C_INF, C_INF, &ev, ctx)
            _undo(&s, &u)

    cdef dict values = {}
    for i in range(n_moves):
        values[_id2c(mv.c[i])] = results[i]
    return values


def get_stats():
    """Diagnostics from the most recent solve call.

    Node attribution (sum to nodes_explored):
      - terminals     : returned at the terminal/early-term check (line 1885)
      - bounds_cuts   : returned via futility (hi<=alpha or lo>=beta)
      - tt_hits       : returned via a TT lookup (exact / lower / upper)
      - no_moves      : returned because mv.n == 0 after legal+cull
      - interior      : did full work — legal+cull+order+children loop+TT store
                        (= nodes_explored − sum of the above)
    """
    cdef long long n  = _g_ctx.nodes
    cdef long long c  = _g_ctx.cuts
    cdef long long th = _g_ctx.tt_hits
    cdef long long ts = _g_ctx.tt_stores
    cdef long long t  = _g_ctx.terminals
    cdef long long b  = _g_ctx.bounds_cuts
    cdef long long nm = _g_ctx.no_moves
    interior = n - t - b - th - nm
    return {
        "nodes_explored": n,
        "cutoffs": c,
        # Force float division — both globals are cdef long long, so plain
        # ``/`` lowers to C integer division and rounds to 0 for any
        # ratio < 1.
        "pruning_ratio": (<double>c / <double>n) if n > 0 else 0.0,
        "tt_hits":   th,
        "tt_stores": ts,
        "terminals":   t,
        "bounds_cuts": b,
        "no_moves":    nm,
        "interior":    interior,
    }
