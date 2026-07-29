// Play — a full game of Ulti against the frontier champion.
//
// Pick a seat, then play the whole thing: the full 33-rung bidding ladder (any
// seat may open; you bid on your turn, the two AI seats bid with the frontier
// net-bidder), an optional kontra round (simple contracts), then the hand is
// played out (AI = frontier PIMC). The play table reuses the same UltiTable as
// the betli tab and animates the AI's cards one at a time. Server is
// authoritative — every action POSTs and the AI's turns come back resolved.

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { api, type PlayState, type PlayLegalBid, type PlayAnalysis, type PlayResult } from "./api";
import type { Card, Suit, Rank } from "./cards";
import { SUIT_HUN, SUIT_SYMBOL } from "./cards";
import { CardBack, CardView } from "./CardView";
import { UltiTable, TalonStrip, type SeatChrome } from "./UltiTable";
import { useStepScrubber } from "./useStepScrubber";
import { PuzzleRush } from "./PuzzleRush";

type Seat = 0 | 1 | 2;

const SEAT_META: Record<Seat, { short: string; flavor: string; accent: string }> = {
  0: { short: "P0 · Forehand", flavor: "You act first in the auction.",
       accent: "linear-gradient(135deg, #6b1f8a 0%, #a23ed1 100%)" },
  1: { short: "P1 · Middle",   flavor: "Second to act.",
       accent: "linear-gradient(135deg, #2b5fa8 0%, #5089d6 100%)" },
  2: { short: "P2 · Rear",     flavor: "Third to act.",
       accent: "linear-gradient(135deg, #2b5fa8 0%, #5089d6 100%)" },
};

const SUIT_ORDER: Record<Suit, number> = { hearts: 0, bells: 1, leaves: 2, acorns: 3 };
const RANK_ORDER: Record<Rank, number> = {
  "7": 0, "8": 1, "9": 2, lower: 3, upper: 4, king: 5, "10": 6, ace: 7,
};
function sortedHand(cards: Card[]): Card[] {
  return [...cards].sort((a, b) => {
    const s = SUIT_ORDER[a.suit] - SUIT_ORDER[b.suit];
    return s !== 0 ? s : RANK_ORDER[a.rank] - RANK_ORDER[b.rank];
  });
}
function placeholderHand(n: number, offset: number): Card[] {
  const out: Card[] = [];
  for (let i = 0; i < n; i++) out.push({ suit: "acorns", rank: "7", id: -(offset + i + 1) });
  return out;
}

const PLAYER_LABEL: Record<number, string> = { 0: "P0", 1: "P1", 2: "P2" };
const TRUMP_LABEL = (s: string | null | undefined): string =>
  s ? `${SUIT_SYMBOL[s as Suit]} ${SUIT_HUN[s as Suit]}` : "színtelen";
const KONTRA_WORD: Record<number, string> = { 0: "", 1: "kontra", 2: "rekontra" };
// Play-index → role name (0 = soloist). Used by the play log and seat labels.
const ROLE_LABEL: Record<number, string> = { 0: "Játékos", 1: "Védő 1", 2: "Védő 2" };
// Compact Hungarian rank labels for the play log.
const RANK_SHORT: Record<Rank, string> = {
  "7": "7", "8": "8", "9": "9", lower: "alsó", upper: "felső", king: "K", "10": "10", ace: "ász",
};
// A move in the play log: the coloured suit icon (as on the trump buttons) + rank.
function CardChip({ card }: { card: Card }) {
  return (
    <span className="play-log-card">
      <span className={`trump-symbol trump-${card.suit}`}>{SUIT_SYMBOL[card.suit]}</span>
      <span className="play-log-rank">{RANK_SHORT[card.rank]}</span>
    </span>
  );
}

// ── Play-phase animation (ported from BetliHu) ──────────────────────────────────
const ANIM_STEP_MS = 850;
const ANIM_TRICK_PAUSE_MS = 1250;

function applyUserPlay(cur: PlayState, card: Card): PlayState {
  const seat = (cur.human_play_index ?? 0) as Seat;
  const ply = cur.history?.length ?? 0;
  const step = { player_id: seat, card, trick_index: Math.floor(ply / 3), trick_position: ply % 3, by_ai: false };
  const trickFull = (cur.current_trick?.length ?? 0) === 3;
  const play = { player_id: seat, card };
  const newTrick = trickFull ? [play] : [...(cur.current_trick ?? []), play];
  const newHandSizes = [...(cur.hand_sizes ?? [])];
  newHandSizes[seat] = Math.max(0, newHandSizes[seat] - 1);
  const newHands = (cur.hands ?? []).map((h, pid) =>
    pid !== seat ? h : h.filter((c) => c === null || c.id !== card.id));
  return {
    ...cur, history: [...(cur.history ?? []), step], current_trick: newTrick,
    hand_sizes: newHandSizes, hands: newHands, current_player: null, legal_card_ids: null,
  };
}

// A fresh trick-1 baseline built from an auction-resolve response, so the AI's
// opening card(s) animate in from an empty table instead of snapping on at once.
// Everyone holds 10 at the start of trick 1 (talon already buried); the human's
// hand is untouched (only AI cards get played before the human's first turn).
function playBaseline(resp: PlayState): PlayState {
  return {
    ...resp,
    phase: "play" as const,
    kontra: null,
    history: [],
    current_trick: [],
    hand_sizes: [10, 10, 10],
    current_player: null,
    legal_card_ids: null,
    terminal: false,
    result: null,
  };
}

function applyStepToVisible(cur: PlayState, target: PlayState): PlayState {
  const nextStep = target.history![cur.history!.length];
  const newHistory = [...cur.history!, nextStep];
  const trickFull = (cur.current_trick?.length ?? 0) === 3;
  const play = { player_id: nextStep.player_id, card: nextStep.card };
  const newTrick = trickFull ? [play] : [...(cur.current_trick ?? []), play];
  const newHandSizes = [...(cur.hand_sizes ?? [])];
  newHandSizes[nextStep.player_id] = Math.max(0, newHandSizes[nextStep.player_id] - 1);
  const seat = (cur.human_play_index ?? 0) as Seat;
  const newHands = (cur.hands ?? []).map((h, pid) =>
    pid !== seat ? h : h.filter((c) => c === null || c.id !== nextStep.card.id));
  const done = newHistory.length === target.history!.length;
  return {
    ...cur, history: newHistory, current_trick: newTrick, hand_sizes: newHandSizes, hands: newHands,
    current_player: done ? target.current_player : null,
    legal_card_ids: done ? target.legal_card_ids : null,
    terminal: done ? target.terminal : false,
    result: done ? target.result : null,
  };
}


// Timed speech bubble for one seat (ported from trickster's useUltiBubble):
// imperative show(text), visible 2200ms then a 400ms fade. Uses oldtawer's
// existing .ulti-speech-bubble / .ulti-bubble-in|out styles.
function useUltiBubble(): { show: (t: string) => void; clear: () => void; node: ReactNode } {
  const [text, setText] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);
  const timer = useRef<number | null>(null);
  const show = useCallback((msg: string) => {
    if (timer.current) window.clearTimeout(timer.current);
    setText(msg); setVisible(true);
    timer.current = window.setTimeout(() => {
      setVisible(false);
      timer.current = window.setTimeout(() => { setText(null); timer.current = null; }, 400);
    }, 2200);
  }, []);
  const clear = useCallback(() => {
    if (timer.current) window.clearTimeout(timer.current);
    setText(null); setVisible(false); timer.current = null;
  }, []);
  useEffect(() => () => { if (timer.current) window.clearTimeout(timer.current); }, []);
  const node = text ? (
    <div className={`ulti-speech-bubble ${visible ? "ulti-bubble-in" : "ulti-bubble-out"}`}>
      <span>{text}</span>
    </div>
  ) : null;
  return { show, clear, node };
}

export function PlayVsAI() {
  const [showPuzzle, setShowPuzzle] = useState(false);   // Villámtalon mini-game overlay

  const [state,   setState]   = useState<PlayState | null>(null);
  const [pending, setPending] = useState<PlayState | null>(null);   // animation target
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  const [selPos,   setSelPos]   = useState<number | null>(null);   // index into legal_bids
  const [selTrump, setSelTrump] = useState<string | null>(null);
  const [discards, setDiscards] = useState<Set<number>>(new Set());
  const [showCard, setShowCard] = useState(false);                 // scorecard popup
  const [showCaptured, setShowCaptured] = useState(false);         // captured-cards popup
  const [kontraSel, setKontraSel] = useState<Set<string>>(new Set());  // per-unit kontra picks

  // Speech bubbles — one per PLAY-INDEX seat (0 = soloist). Server sends
  // {player, text, ply} events (marriage bemondás, kontra/rekontra); each is shown
  // when the VISIBLE play reaches its ply, so it lands as that player's card is played.
  const bubble0 = useUltiBubble();
  const bubble1 = useUltiBubble();
  const bubble2 = useUltiBubble();
  const bubbleOf = useMemo(() => [bubble0, bubble1, bubble2], [bubble0, bubble1, bubble2]);
  const shownBubbles = useRef<Set<string>>(new Set());
  const resetBubbles = useCallback(() => {
    bubble0.clear(); bubble1.clear(); bubble2.clear(); shownBubbles.current.clear();
  }, [bubble0, bubble1, bubble2]);

  useEffect(() => {
    const src = (pending?.bubbles ?? state?.bubbles ?? []);
    if (!src.length) return;
    const visibleLen = state?.history?.length ?? 0;   // how many plays are on screen
    src.forEach((b) => {
      const ply = b.ply ?? 0;
      const key = `${ply}:${b.player}:${b.text}`;
      if (ply < visibleLen && !shownBubbles.current.has(key)) {
        shownBubbles.current.add(key);
        bubbleOf[b.player]?.show(b.text);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state?.history?.length, pending, state?.bubbles]);

  // Match tracking: accumulate each round's GP across "Next round". The human's
  // REAL seat rotates every round (the deal passes around), so we can't key the
  // scoreboard on real seats — we'd smear the human across all three columns.
  // Instead remap each round into STABLE player space: column 0 = You (always),
  // 1 = the opponent downstream of you, 2 = the opponent upstream. You are P0 forever.
  type RoundRow = { contract: string; gp: number[]; soloist: number; winner: string; silents?: string[] };
  const [rounds, setRounds] = useState<RoundRow[]>([]);
  const recordedGame = useRef<string | null>(null);

  // Record a round exactly once when it finishes (a played hand OR an all-pass penalty).
  useEffect(() => {
    const finished = state?.phase === "done" || state?.phase === "passed";
    if (!finished || !state?.result) return;
    if (recordedGame.current === state.game_id) return;
    recordedGame.current = state.game_id;
    const r = state.result;
    const humanSeat = state.seat;                            // seat the human held THIS round
    const solSeat = r.soloist_seat ?? ((humanSeat - (state.human_play_index ?? 0) + 3) % 3);
    // Real seat of stable player p = (p + humanSeat) % 3, so player 0 = the human.
    const gp = [0, 1, 2].map((p) => r.seat_gp[(p + humanSeat) % 3] ?? 0);
    const soloist = (solSeat - humanSeat + 3) % 3;
    const silents = (r.silents ?? []).map((s) => `${s.label} ${s.gp >= 0 ? "+" : ""}${s.gp}`);
    setRounds((rs) => [...rs, { contract: r.contract, gp, soloist, winner: r.winner, silents }]);
  }, [state?.phase, state?.game_id]);   // eslint-disable-line react-hooks/exhaustive-deps

  const matchGp = useMemo(
    () => rounds.reduce((a, r) => [a[0] + (r.gp[0] ?? 0), a[1] + (r.gp[1] ?? 0), a[2] + (r.gp[2] ?? 0)], [0, 0, 0]),
    [rounds],
  );

  const auction = state?.auction;
  const isBidTurn = state?.phase === "bid" && !!auction?.is_human_turn;
  const animating = pending !== null;

  // Clear the per-unit kontra picks whenever a fresh kontra decision surfaces.
  useEffect(() => {
    if (state?.phase === "kontra" && state.kontra?.pending) setKontraSel(new Set());
  }, [state?.phase, state?.kontra?.pending?.play_index, state?.kontra?.pending?.role]);

  // Animation driver — advance one history step per tick toward the target.
  useEffect(() => {
    if (!pending || !state) return;
    if ((state.history?.length ?? 0) >= (pending.history?.length ?? 0)) {
      // Adopt the full authoritative server state (phase may have flipped to
      // "kontra" or "done" mid-animation), but keep the just-shown trick on
      // screen if the server has already cleared it for the next round.
      const adoptTrick = (pending.current_trick?.length ?? 0) > 0 || (state.current_trick?.length ?? 0) !== 3;
      setState({ ...pending, current_trick: adoptTrick ? pending.current_trick : state.current_trick });
      setPending(null);
      return;
    }
    const trickJustDone = (state.current_trick?.length ?? 0) === 3;
    const t = window.setTimeout(
      () => setState(applyStepToVisible(state, pending)),
      trickJustDone ? ANIM_TRICK_PAUSE_MS : ANIM_STEP_MS,
    );
    return () => window.clearTimeout(t);
  }, [state, pending]);

  // Reset the bid panel on each fresh bid turn. Default to the FIRST option —
  // passz (or Kezdés when you're the holder).
  const bidTurnKey = `${state?.game_id}:${auction?.history.length}:${auction?.turn}:${auction?.awaiting_bid}`;
  useEffect(() => {
    if (!isBidTurn) return;
    setDiscards(new Set());
    const bids = auction?.legal_bids ?? [];
    const pos = bids.length ? 0 : null;
    setSelPos(pos);
    const b = pos !== null ? bids[pos] : null;
    setSelTrump(b && b.trump_options.length ? b.trump_options[0] : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bidTurnKey]);

  const selBid: PlayLegalBid | null = useMemo(() => {
    if (!auction?.legal_bids || selPos === null) return null;
    return auction.legal_bids[selPos] ?? null;
  }, [auction, selPos]);

  const onNew = useCallback(async () => {
    setLoading(true); setError(null); setPending(null); resetBubbles();
    try {
      // You always open the first deal as the 12-card holder (seat 0).
      setState(await api.playNew({ seat: 0, seed: undefined }));
    } catch (e) { setError(String(e)); setState(null); }
    finally { setLoading(false); }
  }, [resetBubbles]);

  // Next round rotates who holds the 12 (the opener). A dead deal (all passed)
  // is re-dealt by the same dealer, so pass rotate=false there.
  const onPlayAgain = useCallback(async (rotate = true) => {
    if (!state) return;
    const next = (rotate ? (state.seat + 1) % 3 : state.seat) as Seat;
    const old = state.game_id;
    setLoading(true); setError(null); setPending(null); resetBubbles();
    try { setState(await api.playNew({ seat: next })); api.playDelete(old).catch(() => {}); }
    catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }, [state, resetBubbles]);

  const onAbandon = useCallback(async () => {
    if (state) await api.playDelete(state.game_id).catch(() => {});
    setState(null); setError(null); setPending(null); resetBubbles();
    setRounds([]); recordedGame.current = null;
  }, [state, resetBubbles]);

  // When an auction action resolves straight into play, the server may have
  // already played the AI's opening card(s) before your turn. Animate those in
  // from a fresh baseline instead of snapping the whole trick onto the table.
  const settleInto = useCallback((resp: PlayState) => {
    if ((resp.phase === "play" || resp.phase === "kontra") && (resp.history?.length ?? 0) > 0) {
      setState(playBaseline(resp));
      setPending(resp);
    } else {
      setState(resp);
    }
  }, []);

  // One confirm action for the whole ladder — Passz (bury 2), Kezdés (holder:
  // start play), or a contract bid (discard 2 + trump).
  const onConfirm = useCallback(async () => {
    if (!state || !selBid) return;
    setLoading(true); setError(null);
    try {
      if (selBid.kind === "pass") {
        settleInto(await api.playPass(state.game_id, [...discards]));      // bury 2 as the talon
      } else {
        settleInto(await api.playBid({
          game_id: state.game_id, rung_index: selBid.rung_index, bid_index: selBid.bid_index,
          // Trump is deferred for a plain colored game (chosen after the auction);
          // piros forces hearts (the single option). Colored → send null.
          trump: selBid.trump_options.length === 1 ? selBid.trump_options[0] : null,
          discard_ids: [...discards],
        }));
      }
    } catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }, [state, selBid, selTrump, discards, settleInto]);

  // Auction step (overcall): pick the talon up to bid, or decline without it.
  const onPickup = useCallback(async () => {
    if (!state) return;
    setLoading(true); setError(null);
    try { setState(await api.playPickup(state.game_id)); }   // stays in the bid step
    catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }, [state]);
  const onAuctionPass = useCallback(async () => {
    if (!state) return;
    setLoading(true); setError(null);
    try { settleInto(await api.playPass(state.game_id, [])); }   // decline / accept, no discard
    catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }, [state, settleInto]);
  const onTrump = useCallback(async (suit: string) => {
    if (!state) return;
    setLoading(true); setError(null);
    try { settleInto(await api.playTrump(state.game_id, suit)); }   // you're the soloist → you lead
    catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }, [state, settleInto]);

  // Kontra the given units (empty = pass/"Tovább"). For a combined game each unit is
  // decided separately; the human picks any subset.
  const onKontra = useCallback(async (units: string[]) => {
    if (!state) return;
    setLoading(true); setError(null);
    // Drop the kontra box right away and keep the table as-is, so the continuation
    // cards can animate in one-by-one instead of snapping on all at once.
    const base = { ...state, phase: "play" as const, kontra: null };
    setState(base);
    try {
      const resp = await api.playKontra({ game_id: state.game_id, units });
      if ((resp.history?.length ?? 0) > (base.history?.length ?? 0) && resp.phase !== "bid") {
        setPending(resp);   // animate the cards the AI plays after your decision
      } else {
        setState(resp);
      }
    } catch (e) { setError(String(e)); setState(state); }
    finally { setLoading(false); }
  }, [state]);
  const toggleKontraUnit = useCallback((key: string) => {
    setKontraSel((prev) => {
      const n = new Set(prev);
      if (n.has(key)) n.delete(key); else n.add(key);
      return n;
    });
  }, []);

  const onPlayCard = useCallback(async (card: Card) => {
    if (!state || loading || animating || state.phase !== "play") return;
    if (state.current_player !== state.human_play_index) return;
    if (!state.legal_card_ids?.includes(card.id)) return;
    setState(applyUserPlay(state, card));    // optimistic — drop the card immediately
    setLoading(true); setError(null);
    try { setPending(await api.playMove({ game_id: state.game_id, card_id: card.id })); }
    catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }, [state, loading, animating]);

  const toggleDiscard = useCallback((c: Card) => {
    setDiscards((prev) => {
      const next = new Set(prev);
      if (next.has(c.id)) next.delete(c.id);
      else if (next.size < 2) next.add(c.id);
      return next;
    });
  }, []);

  // ── Post-game analysis board (god solver + branch exploration) ─────────────
  const [analysis, setAnalysis] = useState<PlayAnalysis | null>(null);
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [scrubPly, setScrubPly] = useState(0);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  // Branch: the FULL alternative line (unchanged prefix + god-PV of the chosen fork), held as a
  // complete ply list so branches COMPOSE — you can fork again off any ply of a branch, as deep as
  // you like. `forkPly` is where the latest fork diverged (for the panel + clear).
  const [branch, setBranch] = useState<{ plies: EffectivePly[]; forkPly: number; value: number } | null>(null);
  const [branching, setBranching] = useState(false);

  const onOpenAnalysis = useCallback(async () => {
    if (!state) return;
    setAnalysisLoading(true); setError(null);
    try {
      const ana = await api.playAnalysis({ game_id: state.game_id });
      setAnalysis(ana); setScrubPly(ana.per_ply.length); setBranch(null); setAnalysisOpen(true);
    } catch (e) { setError(String(e)); }
    finally { setAnalysisLoading(false); }
  }, [state]);
  const onCloseAnalysis = useCallback(() => { setAnalysisOpen(false); setBranch(null); }, []);
  const onClearBranch = useCallback(() => {
    if (!branch || !analysis) return;
    setBranch(null); setScrubPly(Math.min(branch.forkPly, analysis.per_ply.length));
  }, [branch, analysis]);

  type EffectivePly = {
    ply_index: number; player_id: 0 | 1 | 2; chosen_card: Card; legal_card_ids: number[];
    verdict: PlayAnalysis["per_ply"][number] | null; by_ai: boolean; is_branch: boolean;
  };
  const effectivePlies = useMemo<EffectivePly[]>(() => {
    if (!analysis) return [];
    if (branch) return branch.plies;                    // the branch already IS the full line
    return analysis.per_ply.map((p, i) => ({
      ply_index: i, player_id: p.player_id, chosen_card: p.chosen_card,
      legal_card_ids: p.legal_card_ids, verdict: p, by_ai: p.by_ai, is_branch: false }));
  }, [analysis, branch]);

  useStepScrubber({ enabled: analysisOpen && analysis !== null, max: effectivePlies.length, setStep: setScrubPly });

  const analysisView = useMemo(() => {
    if (!analysis) return null;
    const hands: Card[][] = analysis.initial_hands.map((h) => [...h]);
    let trick: { player_id: 0 | 1 | 2; card: Card }[] = [];
    for (let i = 0; i < scrubPly && i < effectivePlies.length; i++) {
      const p = effectivePlies[i];
      if (trick.length === 3) trick = [];
      hands[p.player_id] = hands[p.player_id].filter((c) => c.id !== p.chosen_card.id);
      trick.push({ player_id: p.player_id, card: p.chosen_card });
    }
    let activePlayer: 0 | 1 | 2 | null = null;
    let legalIds: Set<number> | null = null;
    let branchAtPly: number | null = null;
    if (scrubPly > 0) {
      const last = effectivePlies[scrubPly - 1];
      activePlayer = last.player_id;
      legalIds = new Set<number>(last.legal_card_ids);
      branchAtPly = scrubPly - 1;
      // Restore the just-played card so the user can click any alternative to fork.
      hands[last.player_id] = [...hands[last.player_id], last.chosen_card];
    }
    return { hands, currentTrick: trick, activePlayer, legalIds, branchAtPly,
             currentPly: scrubPly, thisPly: scrubPly > 0 ? effectivePlies[scrubPly - 1] : null };
  }, [analysis, scrubPly, effectivePlies]);

  const onAnalysisCardClick = useCallback(async (card: Card) => {
    if (!analysis || !analysisView || branching) return;
    const branchAt = analysisView.branchAtPly;
    if (branchAt === null) return;
    const at = effectivePlies[branchAt];
    if (!at.legal_card_ids.includes(card.id)) return;
    setBranching(true); setError(null);
    try {
      // moves = the CURRENT line up to the fork (may itself run through earlier branches), so a
      // fork off a branch replays correctly on the backend.
      const moves = effectivePlies.slice(0, branchAt).map((p) => p.chosen_card.id);
      const resp = await api.pisExplore({
        hands: analysis.initial_hands.map((h) => h.map((c) => c.id)),
        soloist: analysis.soloist, starting_leader: analysis.leader, total_tricks: 10,
        moves, forced_card_id: card.id,
        contract: analysis.solve_contract, build_contract: analysis.build_contract,
        trump: analysis.trump,
        talon: analysis.talon.map((c) => c.id),
        declare_marriages: analysis.declare_marriages,
        marriage_restrict: analysis.marriage_restrict,
        multi_weights: analysis.multi_weights,
      });
      // alt_pv[0] IS the forced card; splice the new god-continuation onto the unchanged prefix →
      // the full line. Keeps ply_index == array position, so it composes for the next fork.
      const prefix = effectivePlies.slice(0, branchAt);
      const pvPlies: EffectivePly[] = resp.alt_pv.map((s, j) => ({
        ply_index: branchAt + j, player_id: s.player_id as 0 | 1 | 2, chosen_card: s.card,
        legal_card_ids: s.legal_card_ids, verdict: null, by_ai: false, is_branch: true }));
      setBranch({ plies: [...prefix, ...pvPlies], forkPly: branchAt, value: resp.value });
      setScrubPly(branchAt + 1);
    } catch (e) { setError(String(e)); }
    finally { setBranching(false); }
  }, [analysis, analysisView, effectivePlies, branching]);

  // Shared end-of-hand result — one simple line ("Nyertél 4 pontot") + contract + the
  // action buttons. Used identically by a played game AND the all-pass screen.
  const renderResult = (r: PlayResult, withAnalysis: boolean) => (
    <div className={`play-side-inner play-side-result ${r.user_won ? "is-win" : "is-loss"}`}>
      <div className="betli-hu-result-headline">
        {r.user_won ? "Nyertél" : "Vesztettél"} {Math.abs(r.human_gp)} pontot
      </div>
      <div className="betli-hu-result-sub">
        {r.contract}{r.kontra_level > 0 ? ` · ${KONTRA_WORD[r.kontra_level]}` : ""}
      </div>
      {(r.silents?.length ?? 0) > 0 && (
        <div className="play-silents" style={{ marginTop: 8 }}>
          <span className="play-silents-label">csendes játékok:</span>
          {r.silents!.map((s, i) => (
            <span key={i} className="play-silent-item">{s.label} {s.gp >= 0 ? "+" : ""}{s.gp}</span>
          ))}
        </div>
      )}
      <div className="play-side-actions">
        <button className="betli-hu-deal-btn betli-hu-deal-btn--sm" onClick={() => onPlayAgain(true)} disabled={loading}>
          {loading ? "Osztás…" : "Következő"}
        </button>
        {withAnalysis && (
          <button className="btn" onClick={onOpenAnalysis} disabled={analysisLoading}>
            {analysisLoading ? "Elemzés…" : "Elemzés"}
          </button>
        )}
        {rounds.length > 0 && (
          <button className="btn" onClick={() => setShowCard(true)} disabled={loading}>Pontszámok</button>
        )}
        <button className="btn" onClick={onAbandon} disabled={loading}>Új játszma</button>
      </div>
    </div>
  );

  // ── Setup ────────────────────────────────────────────────────────────────
  if (showPuzzle) {
    return <PuzzleRush onExit={() => setShowPuzzle(false)} />;
  }

  if (!state) {
    return (
      <div className="app betli-hu-splash">
        <main className="main betli-hu-splash-main">
          <section style={{ textAlign: "center" }}>
            <h1 className="betli-hu-title">Ulti</h1>
            <div className="splash-actions">
              <button className="betli-hu-deal-btn" onClick={onNew} disabled={loading}>
                {loading ? "…" : "Új játék"}
              </button>
              <button className="betli-hu-deal-btn betli-hu-deal-btn--ghost"
                      onClick={() => setShowPuzzle(true)}>
                Villámtalon
              </button>
            </div>
            {error && <div className="error" style={{ marginTop: 14, maxWidth: 480, margin: "14px auto 0" }}>{error}</div>}
          </section>
        </main>
      </div>
    );
  }

  // ── Post-game analysis overlay (god solver + branch exploration) ────────────
  if (analysisOpen && analysis && analysisView) {
    const a = analysisView;
    const v = a.thisPly;
    const maxPly = effectivePlies.length;
    const anaHpi = (analysis.human_play_index ?? 0) as Seat;
    const anaSeat = (pid: Seat): SeatChrome => ({   // same chrome as the game phase (roleTag)
      label: (
        <>
          <span className={`ulti-role-tag ${pid === 0 ? "ulti-role-soloist" : "ulti-role-defender"}`}>
            {pid === 0 ? "Játékos" : "Védő"}
          </span>
          {pid === anaHpi && <span className="ulti-role-tag" style={{ background: "#3a6", color: "#fff" }}>te</span>}
        </>
      ),
    });
    const anaSeats = { 0: anaSeat(0), 1: anaSeat(1), 2: anaSeat(2) };
    const fmt = (n: number) => (Number.isInteger(n) ? String(n) : n.toFixed(1));
    return (
      <div className="app betli-hu-game play-vs-ai">
        <header className="topbar">
          <div className="brand">
            <div className="title">
              Elemzés — {analysis.contract}
              {analysis.trump && <span className="muted"> · {TRUMP_LABEL(analysis.trump)}</span>}
              {branch && <span className="ulti-role-tag" style={{ background: "#a23ed1", color: "#fff", marginLeft: 8 }}>alt ág</span>}
            </div>
            <div className="subtitle">
              {a.currentPly}. / {maxPly}. lépés · <span className="kbd">←</span> <span className="kbd">→</span> a léptetéshez · kattints egy lapra egy ág kipróbálásához
              {v?.verdict?.is_blunder && <> · <b style={{ color: "#ff8a8a" }}>HIBA</b></>}
              {branching && <> · <i>számolás…</i></>}
            </div>
          </div>
          <div className="controls">
            <button className="btn" onClick={() => setScrubPly((p) => Math.max(0, p - 1))} disabled={a.currentPly === 0}>←</button>
            <button className="btn" onClick={() => setScrubPly((p) => Math.min(maxPly, p + 1))} disabled={a.currentPly >= maxPly}>→</button>
            {branch && <button className="btn" onClick={onClearBranch}>Ág törlése</button>}
            <button className="btn btn-primary" onClick={onCloseAnalysis}>Vissza</button>
          </div>
        </header>
        <main className="main">
          <section>
            {error && <div className="error">{error}</div>}
            <UltiTable
              hands={[sortedHand(a.hands[0]), sortedHand(a.hands[1]), sortedHand(a.hands[2])]}
              seats={anaSeats}
              currentTrick={a.currentTrick}
              activePlayer={a.activePlayer}
              legalIds={a.legalIds}
              onCardClick={branching ? undefined : onAnalysisCardClick}
              seatNames={{ 0: "Játékos", 1: "Védő", 2: "Védő" }}
              bottomSeat={anaHpi}
              aboveDefenders={
                <>
                  {analysis.trump && (
                    <div className="row" style={{ justifyContent: "center", marginBottom: 4 }}>
                      <span className={`play-badge ${analysis.trump === "hearts" ? "is-piros" : ""}`}>
                        Adu: {TRUMP_LABEL(analysis.trump)}
                      </span>
                    </div>
                  )}
                  <TalonStrip talon={analysis.talon} />
                </>
              }
            />
            {(v?.verdict || branch) && (
              <div className="panel" style={{ marginTop: 10 }}>
                {v?.verdict ? (
                  <>
                    <div className="panel-title">Isteni ítélet — {v.ply_index + 1}. lépés</div>
                    <div className="row" style={{ gap: 16, flexWrap: "wrap", fontSize: 13 }}>
                      <div>
                        <div className="muted" style={{ fontSize: 11 }}>{v.verdict.player_id === 0 ? "Játékos" : "Védő"} lépett</div>
                        <div><CardChip card={v.verdict.chosen_card} /> · érték {fmt(v.verdict.god_chosen_value)}</div>
                      </div>
                      <div>
                        <div className="muted" style={{ fontSize: 11 }}>Isteni választás</div>
                        <div><CardChip card={v.verdict.god_best_card} /> · érték {fmt(v.verdict.god_best_value)}</div>
                      </div>
                      <div>
                        <div className="muted" style={{ fontSize: 11 }}>Ítélet</div>
                        <div style={{ color: v.verdict.is_blunder ? "#ff8a8a" : "#9fe3a5", fontWeight: 600 }}>
                          {v.verdict.is_blunder ? "Hiba" : "OK"}
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="panel-title">Alternatív ág · isteni folytatás</div>
                    <div className="muted" style={{ fontSize: 12 }}>
                      Elágazás a(z) {(branch?.forkPly ?? 0) + 1}. lépésnél — innentől minden lépés isteni-optimális. Ág végérték: {fmt(branch?.value ?? 0)}.
                    </div>
                  </>
                )}
              </div>
            )}
          </section>
          <section>
            <div className="panel betli-hu-log-panel">
              <div className="panel-title">Lépések · isteni ítélet</div>
              <div className="betli-hu-log-scroll">
                <table className="play-sc-table" style={{ fontSize: 12, width: "100%" }}>
                  <thead>
                    <tr><th>#</th><th>P</th><th>Lépett</th><th>Isteni</th><th>Ítélet</th></tr>
                  </thead>
                  <tbody>
                    {effectivePlies.map((p) => {
                      const selected = scrubPly - 1 === p.ply_index;
                      const blunder = p.verdict?.is_blunder ?? false;
                      return (
                        <tr key={`${p.ply_index}-${p.is_branch ? "b" : "o"}`}
                            onClick={() => setScrubPly(p.ply_index + 1)}
                            style={{ cursor: "pointer",
                              background: selected ? "rgba(99,200,255,0.16)" : p.is_branch ? "rgba(162,62,209,0.10)" : undefined,
                              color: blunder ? "#ff8a8a" : undefined, fontWeight: blunder ? 600 : undefined }}>
                          <td className="muted">{p.ply_index + 1}</td>
                          <td>{p.player_id === 0 ? "J" : `V${p.player_id}`}{p.by_ai ? "·AI" : ""}{p.is_branch ? "·alt" : ""}</td>
                          <td><CardChip card={p.chosen_card} /></td>
                          <td>{p.verdict ? <CardChip card={p.verdict.god_best_card} /> : "—"}</td>
                          <td>{p.verdict ? (blunder ? "hiba" : "ok") : "(ág)"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        </main>
      </div>
    );
  }

  // ── All-passed — SAME shell + result box as a played game (just no table/play). ──
  if (state.phase === "passed") {
    const r = state.result ?? null;
    return (
      <div className="app betli-hu-game play-vs-ai">
        <main className="main">
          <section>
            {error && <div className="error">{error}</div>}
            <div className="play-passed-note">Ebben a leosztásban senki sem licitált.</div>
          </section>
          <section>
            <div className="play-side-box">{r ? renderResult(r, false) : null}</div>
          </section>
        </main>

        {showCard && (
          <div className="play-modal-backdrop" onClick={() => setShowCard(false)}>
            <div className="play-modal" onClick={(e) => e.stopPropagation()}>
              <div className="play-modal-head">
                <span>Pontszámok ({rounds.length} kör)</span>
                <button className="btn" onClick={() => setShowCard(false)}>×</button>
              </div>
              <table className="play-sc-table">
                <thead><tr><th>#</th><th>Játék</th><th>Te</th><th>Gép 1</th><th>Gép 2</th></tr></thead>
                <tbody>
                  {rounds.map((rr, i) => (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td>{rr.contract}{(rr.silents?.length ?? 0) > 0 &&
                        <span className="play-sc-silent"> · {rr.silents!.join(", ")}</span>}</td>
                      {([0, 1, 2] as const).map((s) => (
                        <td key={s} className={`${rr.gp[s] > 0 ? "play-sc-pos" : rr.gp[s] < 0 ? "play-sc-neg" : ""} ${s === 0 ? "play-sc-you" : ""}`}>
                          {rr.gp[s] > 0 ? "+" : ""}{rr.gp[s]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr>
                    <td></td><td>Összesen</td>
                    {([0, 1, 2] as const).map((s) => (
                      <td key={s} className={`${matchGp[s] > 0 ? "play-sc-pos" : matchGp[s] < 0 ? "play-sc-neg" : ""} ${s === 0 ? "play-sc-you" : ""}`}>
                        {matchGp[s] > 0 ? "+" : ""}{matchGp[s]}
                      </td>
                    ))}
                  </tr>
                </tfoot>
              </table>
            </div>
          </div>
        )}
      </div>
    );
  }

  // ── Trump select — you won a plain colored game; declare the suit before play ──
  if (state.phase === "trump_select") {
    const opts = state.trump_options ?? ["acorns", "leaves", "bells"];
    const handCards = sortedHand((state.own_hand ?? []) as Card[]);
    const meSeat = state.seat as Seat;
    const tHands: [Card[], Card[], Card[]] = [[], [], []];
    for (let s = 0 as Seat; s <= 2; s = (s + 1) as Seat) {
      tHands[s] = s === meSeat ? handCards : placeholderHand(10, s * 100);
    }
    const tHidden = new Set<0 | 1 | 2>(([0, 1, 2] as Seat[]).filter((s) => s !== meSeat));
    const tChrome = (pid: Seat): SeatChrome => ({ label: <>P{pid}{pid === meSeat ? " (te)" : ""}</> });
    const tSeats = { 0: tChrome(0), 1: tChrome(1), 2: tChrome(2) };
    const trumpPanel = (
      <div className="ulti-auction-overlay">
        <div className="ulti-auction-title">Adu választás</div>
        <div className="ulti-auction-current">
          Megnyerted: <b>{state.contract}</b> — válaszd meg az adu színt.
        </div>
        <div className="ulti-trump-buttons">
          {opts.map((s) => (
            <button key={s} className={`ulti-trump-btn trump-${s}`} onClick={() => onTrump(s)} disabled={loading}>
              <span className="ulti-trump-symbol">{SUIT_SYMBOL[s as Suit]}</span> {SUIT_HUN[s as Suit]}
            </button>
          ))}
        </div>
      </div>
    );
    return (
      <div className="app betli-hu-game play-vs-ai">
        <main className="main">
          <section>
            {error && <div className="error">{error}</div>}
            <UltiTable
              hands={tHands} seats={tSeats} currentTrick={[]} activePlayer={null} legalIds={null}
              seatNames={{ 0: "P0", 1: "P1", 2: "P2" }} hiddenSeats={tHidden}
              bottomSeat={meSeat} hideTrick belowTrick={trumpPanel}
            />
          </section>
        </main>
      </div>
    );
  }

  // ── Bidding — SAME table as the play phase (UltiTable): opponents face-down
  //    top-left/right, your hand fanned at the bottom. One ladder dropdown
  //    (Passz / Kezdés / a contract) + one confirm button in the trick zone. ──
  if (state.phase === "bid" && auction) {
    const isHolder = !!auction.is_holder;          // you own the standing bid
    const reclaim = !!auction.reclaim;             // forehand's last look after all-pass
    const awaitingBid = !!auction.awaiting_bid;    // bid step: you hold 12
    const canPickup = !!auction.can_pickup;        // auction step: may Felveszem
    const handCards = awaitingBid
      ? sortedHand((state.bid_hand ?? []) as Card[])        // 12 — bid step
      : sortedHand((state.own_hand ?? []) as Card[]);       // 10 — auction step
    const talonSet = new Set(state.talon_ids ?? []);
    const showTalon = !!auction.picked_up;         // ring the 2 cards you just took up
    const hist = auction.history.slice(-4);
    const meSeat = state.seat as Seat;

    const kind = selBid?.kind ?? "bid";
    const needsDiscard = awaitingBid;                       // bid step always buries/discards 2
    const canConfirm = !!selBid && discards.size === 2;
    const confirmLabel = kind === "pass" ? "Passzolok" : isHolder ? "Emelek" : "Licitálok";

    // Same table as play — opponents face down top-left/right, your hand fanned at
    // the bottom — but the central trick "playing area" is hidden until play starts.
    const bidHands: [Card[], Card[], Card[]] = [[], [], []];
    for (let s = 0 as Seat; s <= 2; s = (s + 1) as Seat) {
      bidHands[s] = s === meSeat ? handCards : placeholderHand(10, s * 100);
    }
    const bidHidden = new Set<0 | 1 | 2>(([0, 1, 2] as Seat[]).filter((s) => s !== meSeat));
    const bidChrome = (pid: Seat): SeatChrome => ({
      label: (
        <>
          P{pid}{pid === meSeat ? " (te)" : ""}
          {auction.current?.pid === pid && <span className="ulti-role-tag ulti-role-soloist"> licitál</span>}
        </>
      ),
    });
    const bidSeats = { 0: bidChrome(0), 1: bidChrome(1), 2: bidChrome(2) };

    const auctionPanel = (
      <div className="ulti-auction-overlay">
        <div className="ulti-auction-title">Licitálás</div>
        {auction.current && (
          <div className="ulti-auction-current">
            <span className="ulti-auction-label">Aktuális licit:</span>{" "}
            <span className="ulti-auction-bid-text">{auction.current.contract}</span>
            {/* Adu color is hidden during the auction — announced only when play begins. */}
            <span className="ulti-auction-holder"> — {PLAYER_LABEL[auction.current.pid]}{auction.current.pid === state.seat ? " (te)" : ""}</span>
          </div>
        )}

        {hist.length > 0 && (
          <div className="ulti-auction-history">
            {hist.map((h, i) => (
              <div key={i} className="ulti-auction-history-entry">
                <span className="ulti-auction-player">{PLAYER_LABEL[h.pid]}{h.pid === state.seat ? " (te)" : ""}</span>
                {h.kind === "pass"
                  ? <span className="ulti-auction-action pass">passz</span>
                  : <span className="ulti-auction-action bid">{h.contract}{h.trump ? ` · ${TRUMP_LABEL(h.trump)}` : ""}</span>}
              </div>
            ))}
          </div>
        )}

        {awaitingBid ? (
          /* BID STEP — you hold 12: pick from the ladder + discard 2 */
          <>
            <div className="ulti-bid-phase-info">
              {kind === "pass"
                ? <>Dobj el két lapot a talonba. <b>{discards.size}/2</b></>
                : <>Válassz, és dobj el két lapot. <b>{discards.size}/2</b></>}
            </div>
            <div className="ulti-bid-select-row">
              <select className="ulti-bid-select" value={selPos ?? ""}
                      onChange={(e) => {
                        const pos = Number(e.target.value); setSelPos(pos);
                        const b = auction.legal_bids?.[pos];
                        setSelTrump(b && b.trump_options.length ? b.trump_options[0] : null);
                      }}>
                {(auction.legal_bids ?? []).map((b, i) => (
                  <option key={`${b.kind}:${b.rung_index}:${b.bid_index}`} value={i}>
                    {b.label}{b.kind === "bid" ? ` — ${b.value}p` : ""}
                  </option>
                ))}
              </select>
              {/* No trump buttons here — the color is declared AFTER the auction
                  (Ulti keeps it hidden during bidding). Piros = hearts, colorless = none. */}
              {selBid && kind === "bid" && selBid.trump_options.length === 1 && (
                <span className="ulti-badge is-piros">adu: {TRUMP_LABEL(selBid.trump_options[0])}</span>
              )}
              {selBid && kind === "bid" && selBid.colorless && <span className="ulti-badge">színtelen</span>}
              <button className="ulti-bid-confirm" onClick={onConfirm} disabled={loading || !canConfirm}>
                {loading ? "…" : confirmLabel}
              </button>
            </div>
          </>
        ) : (
          /* AUCTION STEP — you hold 10: pick the talon up to bid, or decline */
          <>
            <div className="ulti-bid-phase-info">
              {reclaim
                ? "Vedd fel a talont és játssz, vagy passzolj."
                : isHolder
                ? "Kezdheted a játékot, vagy emelj."
                : canPickup
                ? "Vedd fel a talont, vagy passzolj."
                : "Passzolsz?"}
            </div>
            <div className="ulti-bid-bottom-row">
              <button className="ulti-bid-accept" onClick={onAuctionPass} disabled={loading}>
                {isHolder ? "Elfogadom" : "Passz"}
              </button>
              {canPickup && (
                <button className="ulti-bid-felvesz" onClick={onPickup} disabled={loading}>Felveszem</button>
              )}
            </div>
          </>
        )}
      </div>
    );

    return (
      <div className="app betli-hu-game play-vs-ai">
        <main className="main">
          <section>
            {error && <div className="error">{error}</div>}
            <UltiTable
              hands={bidHands}
              seats={bidSeats}
              currentTrick={[]}
              activePlayer={needsDiscard ? meSeat : null}
              legalIds={null}
              onCardClick={needsDiscard ? toggleDiscard : undefined}
              selectedIds={needsDiscard ? discards : undefined}
              markIds={needsDiscard && showTalon ? talonSet : undefined}
              seatNames={{ 0: "P0", 1: "P1", 2: "P2" }}
              hiddenSeats={bidHidden}
              bottomSeat={meSeat}
              hideTrick
              belowTrick={auctionPanel}
            />
          </section>
        </main>
      </div>
    );
  }

  // ── Play / Kontra / Done ────────────────────────────────────────────────────
  const hpi = (state.human_play_index ?? 0) as Seat;
  const terminal = !!state.terminal;
  const result = state.result ?? null;
  const kontra = state.kontra ?? null;
  const inKontra = state.phase === "kontra" && !!kontra?.is_human_turn;
  const kLevel = state.kontra_level ?? 0;
  const isMyTurn = state.phase === "play" && state.current_player === hpi && !terminal && !animating;

  // terített: after trick 1 the soloist (play-index 0) lays their hand face-up for the defenders.
  const revealSol = !!state.reveal_soloist;
  const tableHands: [Card[], Card[], Card[]] = [[], [], []];
  for (let s = 0 as Seat; s <= 2; s = (s + 1) as Seat) {
    if (s === hpi || (revealSol && s === 0)) {
      tableHands[s] = sortedHand((state.hands?.[s] ?? []).filter((c): c is Card => c !== null));
    } else {
      tableHands[s] = placeholderHand(state.hand_sizes?.[s] ?? 0, s * 100);
    }
  }
  const hiddenSeats = new Set<0 | 1 | 2>(
    ([0, 1, 2] as Seat[]).filter((s) => s !== hpi && !(revealSol && s === 0)));
  const roleTag = (pid: Seat): SeatChrome => ({
    label: (
      <>
        <span className={`ulti-role-tag ${pid === 0 ? "ulti-role-soloist" : "ulti-role-defender"}`}>
          {pid === 0 ? "Játékos" : "Védő"}
        </span>
        {pid === hpi && <span className="ulti-role-tag" style={{ background: "#3a6", color: "#fff" }}>te</span>}
      </>
    ),
    bubbleNode: bubbleOf[pid].node,   // marriage / kontra speech bubble (play-index)
  });
  const seats = { 0: roleTag(0), 1: roleTag(1), 2: roleTag(2) };
  const legalSet = new Set<number>(state.legal_card_ids ?? []);

  const kontraTag = kLevel > 0
    ? <span className="ulti-role-tag" style={{ background: "#d23552", color: "#fff", marginLeft: 6 }}>{KONTRA_WORD[kLevel]}</span>
    : null;

  // The talon: 2 cards face-down beside the table during play, flipped up at the end.
  const talonCount = state.talon_count ?? 2;
  const talonCards = terminal ? (state.talon ?? null) : null;
  const talonNode = (
    <div className="play-talon-pile">
      <span className="play-talon-label">Talon</span>
      <div className="play-talon-cards">
        {talonCards
          ? talonCards.map((c) => <CardView key={c.id} card={c} size="thumb" />)
          : Array.from({ length: talonCount }).map((_, i) => <CardBack key={i} size="thumb" />)}
      </div>
    </div>
  );

  // The side box above the Játékmenet log: hosts the kontra decision, then the
  // end-of-hand result — instead of a bottom-floating popup. Empty when idle.
  const sideBox = (
    <div className="play-side-box">
      {inKontra && kontra?.pending ? (
        <div className="play-side-inner">
          <div className="play-side-title">{kontra.role === "sol" ? "Rekontra döntés" : "Kontra döntés"}</div>
          {(kontra.units?.length ?? 0) <= 1 ? (
            /* Simple game — one unit: direct kontra / pass (as before). */
            <>
              <div className="play-kontra-text">
                {kontra.role === "sol"
                  ? <>Kontráztak: <b>{kontra.units?.[0]?.label ?? kontra.primary}</b> ({state.contract}) — rekontrázol?</>
                  : <>Kontrázod: <b>{kontra.units?.[0]?.label ?? kontra.primary}</b> ({state.contract}{state.trump ? `, ${TRUMP_LABEL(state.trump)}` : ""})?</>}
              </div>
              <div className="row" style={{ gap: 8, marginTop: 10 }}>
                <button className="btn btn-primary" onClick={() => onKontra((kontra.units ?? []).map((u) => u.key))}
                        disabled={loading} style={{ background: "#d23552", borderColor: "#d23552" }}>
                  {kontra.role === "sol" ? "Rekontra" : "Kontra"}
                </button>
                <button className="btn" onClick={() => onKontra([])} disabled={loading}>Tovább</button>
              </div>
            </>
          ) : (
            /* Combined game — pick which units to (re)kontra separately. */
            <>
              <div className="play-kontra-text">
                {kontra.role === "sol"
                  ? <>Melyik egységet rekontrázod? ({state.contract})</>
                  : <>Mit kontrázol? — több is választható ({state.contract}{state.trump ? `, ${TRUMP_LABEL(state.trump)}` : ""})</>}
              </div>
              <div className="play-kontra-units">
                {(kontra.units ?? []).map((u) => (
                  <button key={u.key} type="button" disabled={loading}
                          className={`play-kontra-unit ${kontraSel.has(u.key) ? "is-sel" : ""}`}
                          onClick={() => toggleKontraUnit(u.key)}>
                    {u.label}
                  </button>
                ))}
              </div>
              <div className="row" style={{ gap: 8, marginTop: 10 }}>
                <button className="btn btn-primary" onClick={() => onKontra([...kontraSel])}
                        disabled={loading || kontraSel.size === 0}
                        style={{ background: "#d23552", borderColor: "#d23552" }}>
                  {kontra.role === "sol" ? "Rekontra" : "Kontra"}{kontraSel.size > 0 ? ` (${kontraSel.size})` : ""}
                </button>
                <button className="btn" onClick={() => onKontra([])} disabled={loading}>Tovább</button>
              </div>
            </>
          )}
        </div>
      ) : terminal && result ? (
        renderResult(result, true)
      ) : (
        <div className="play-side-idle">
          {state.phase === "play" && !terminal
            ? (animating || loading ? "A gép lép…" : isMyTurn ? "Te jössz" : "A gép lép…")
            : "Ulti vs AI"}
        </div>
      )}
    </div>
  );

  return (
    <div className="app betli-hu-game play-vs-ai">
      <main className="main">
        <section className="play-col-main">
          {error && <div className="error">{error}</div>}

          <UltiTable
            hands={tableHands}
            seats={seats}
            currentTrick={(state.current_trick ?? []) as { player_id: 0 | 1 | 2; card: Card }[]}
            activePlayer={terminal || state.phase === "kontra" ? null : (state.current_player ?? null)}
            legalIds={isMyTurn ? legalSet : null}
            onCardClick={isMyTurn ? onPlayCard : undefined}
            seatNames={{ 0: "Játékos", 1: "Védő", 2: "Védő" }}
            hiddenSeats={hiddenSeats}
            bottomSeat={hpi}
            aboveDefenders={talonNode}
          />

          {/* Your captured (won) tricks — face down, in groups of 3. Click to reveal. */}
          {(state.captured?.length ?? 0) >= 3 && (() => {
            const nTricks = Math.floor((state.captured?.length ?? 0) / 3);
            return (
              <div className="play-captured" onClick={() => setShowCaptured(true)}
                   title="Kattints a lapok megtekintéséhez">
                <span className="play-captured-label">Ütött lapok ({nTricks})</span>
                {Array.from({ length: nTricks }).map((_, g) => (
                  <div key={g} className="play-captured-trick">
                    {[0, 1, 2].map((i) => <CardBack key={i} size="micro" />)}
                  </div>
                ))}
              </div>
            );
          })()}
        </section>

        <section>
          {/* Persistent contract + live score (moved off the table so it never jumps). The
              marriage bonus (40/20) is withheld until it's declared: the soloist up front,
              each defender at their first card — same timing as the bemondás bubbles. */}
          <div className="panel play-info-panel">
            <div className="play-info-contract">
              <span className={`play-badge ${state.trump === "hearts" ? "is-piros" : ""} ${(state.contract ?? "").includes("betli") ? "is-betli" : ""}`}>
                {state.contract}{state.trump && <> · {TRUMP_LABEL(state.trump)}</>}
                {kLevel > 0 && <span className="play-badge-kontra"> · {KONTRA_WORD[kLevel]}</span>}
              </span>
              {state.contract_value != null && <span className="play-badge-value">{state.contract_value}p</span>}
            </div>
            {state.score && (() => {
              const sc = state.score; const vis = state.history?.length ?? 0;
              const solDisp = (sc.sol_card ?? sc.sol_points) + (sc.marr?.[0] ?? 0);
              const defDisp = (sc.def_card ?? sc.def_points)
                + (vis > 1 ? (sc.marr?.[1] ?? 0) : 0) + (vis > 2 ? (sc.marr?.[2] ?? 0) : 0);
              return (
                <div className="play-info-scores">
                  <span className={hpi === 0 ? "play-score-me" : ""}>Játékos <b>{solDisp}</b></span>
                  <span className="play-score-sep">·</span>
                  <span className={hpi !== 0 ? "play-score-me" : ""}>Védők <b>{defDisp}</b></span>
                </div>
              );
            })()}
          </div>
          {sideBox}
          <div className="panel betli-hu-log-panel">
            <div className="panel-title">Játékmenet</div>
            <div className="betli-hu-log-scroll play-log">
              {Array.from({ length: Math.ceil((state.history?.length ?? 0) / 3) }).map((_, t) => (
                <div key={t} className="play-log-trick">
                  <div className="play-log-trick-head">{t + 1}. ütés</div>
                  {(state.history ?? []).slice(t * 3, t * 3 + 3).map((h, i) => (
                    <div key={i} className="play-log-move">
                      <span className={`play-log-player ${h.player_id === hpi ? "is-me" : ""}`}>
                        {ROLE_LABEL[h.player_id]}{h.player_id === hpi ? " (te)" : ""}
                      </span>
                      <CardChip card={h.card} />
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      {/* Captured cards — revealed grouped by trick */}
      {showCaptured && (
        <div className="play-modal-backdrop" onClick={() => setShowCaptured(false)}>
          <div className="play-modal" onClick={(e) => e.stopPropagation()}>
            <div className="play-modal-head">
              <span>Ütött lapok</span>
              <button className="btn" onClick={() => setShowCaptured(false)}>×</button>
            </div>
            <div className="play-captured-reveal">
              {Array.from({ length: Math.floor((state.captured?.length ?? 0) / 3) }).map((_, g) => (
                <div key={g} className="play-captured-trick-row">
                  {(state.captured ?? []).slice(g * 3, g * 3 + 3).map((c) => (
                    <CardView key={c.id} card={c} size="thumb" />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {showCard && (
        <div className="play-modal-backdrop" onClick={() => setShowCard(false)}>
          <div className="play-modal" onClick={(e) => e.stopPropagation()}>
            <div className="play-modal-head">
              <span>Pontszámok ({rounds.length} kör)</span>
              <button className="btn" onClick={() => setShowCard(false)}>×</button>
            </div>
            {rounds.length === 0 ? (
              <div className="muted">Még nincs lejátszott kör.</div>
            ) : (
              <table className="play-sc-table">
                <thead>
                  <tr><th>#</th><th>Játék</th><th>Te</th><th>Gép 1</th><th>Gép 2</th></tr>
                </thead>
                <tbody>
                  {rounds.map((r, i) => (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td>{r.contract}{(r.silents?.length ?? 0) > 0 &&
                        <span className="play-sc-silent"> · {r.silents!.join(", ")}</span>}</td>
                      {([0, 1, 2] as const).map((s) => (
                        <td key={s} className={`${r.gp[s] > 0 ? "play-sc-pos" : r.gp[s] < 0 ? "play-sc-neg" : ""} ${s === 0 ? "play-sc-you" : ""}`}>
                          {r.gp[s] > 0 ? "+" : ""}{r.gp[s]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr>
                    <td></td><td>Összesen</td>
                    {([0, 1, 2] as const).map((s) => (
                      <td key={s} className={`${matchGp[s] > 0 ? "play-sc-pos" : matchGp[s] < 0 ? "play-sc-neg" : ""} ${s === 0 ? "play-sc-you" : ""}`}>
                        {matchGp[s] > 0 ? "+" : ""}{matchGp[s]}
                      </td>
                    ))}
                  </tr>
                </tfoot>
              </table>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
