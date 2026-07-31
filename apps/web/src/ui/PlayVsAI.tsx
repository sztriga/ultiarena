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
import { AnalysisBoard, AuctionPanel, KontraBox, ResultPanel, Splash } from "./playPanels";

type Seat = 0 | 1 | 2;

import {
  SEAT_META, PLAYER_LABEL, TRUMP_LABEL, KONTRA_WORD, ROLE_LABEL, RANK_SHORT,
  ANIM_STEP_MS, ANIM_TRICK_PAUSE_MS, placeholderHand, CardChip,
  applyUserPlay, playBaseline, applyStepToVisible, useUltiBubble,
  EffectivePly, type AnalysisView,
} from "./playChrome";

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
    // Track what each player has played, then derive hands by FILTERING the initial
    // hands the API sent. Filtering preserves order, so the server's card order (the
    // one rule, ulti.card.sort_hand) survives scrubbing — nothing is re-sorted here.
    const played: Set<number>[] = [new Set(), new Set(), new Set()];
    let trick: { player_id: 0 | 1 | 2; card: Card }[] = [];
    for (let i = 0; i < scrubPly && i < effectivePlies.length; i++) {
      const p = effectivePlies[i];
      if (trick.length === 3) trick = [];
      played[p.player_id].add(p.chosen_card.id);
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
      // Keep the just-played card in hand so the user can click an alternative to fork.
      played[last.player_id].delete(last.chosen_card.id);
    }
    const hands: Card[][] = analysis.initial_hands.map(
      (h, pid) => h.filter((c) => !played[pid].has(c.id)));
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
    <ResultPanel r={r} withAnalysis={withAnalysis} loading={loading}
                 analysisLoading={analysisLoading} hasRounds={rounds.length > 0}
                 onPlayAgain={onPlayAgain} onOpenAnalysis={onOpenAnalysis}
                 onShowCard={() => setShowCard(true)} onAbandon={onAbandon} />
  );

  // ── Setup ────────────────────────────────────────────────────────────────
  if (showPuzzle) {
    return <PuzzleRush onExit={() => setShowPuzzle(false)} />;
  }

  if (!state) {
    return <Splash loading={loading} error={error}
                   onNew={onNew} onPuzzle={() => setShowPuzzle(true)} />;
  }

  // ── Post-game analysis overlay (god solver + branch exploration) ────────────
  if (analysisOpen && analysis && analysisView) {
    return (
      <AnalysisBoard analysis={analysis} analysisView={analysisView}
                     effectivePlies={effectivePlies} branch={branch}
                     branching={branching} error={error}
                     scrubPly={scrubPly} setScrubPly={setScrubPly}
                     onAnalysisCardClick={branching ? undefined : onAnalysisCardClick}
                     onClearBranch={onClearBranch} onCloseAnalysis={onCloseAnalysis} />
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
    const handCards = (state.own_hand ?? []) as Card[];
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
      ? ((state.bid_hand ?? []) as Card[])                  // 12 — bid step
      : ((state.own_hand ?? []) as Card[]);                 // 10 — auction step
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
      <AuctionPanel auction={auction} seat={state.seat} selPos={selPos}
                    setSelPos={setSelPos} setSelTrump={setSelTrump} selBid={selBid}
                    discards={discards} canConfirm={canConfirm} loading={loading}
                    onConfirm={onConfirm} onAuctionPass={onAuctionPass} onPickup={onPickup} />
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
      tableHands[s] = (state.hands?.[s] ?? []).filter((c): c is Card => c !== null);
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
        <KontraBox kontra={kontra} contract={state.contract} trump={state.trump}
                   loading={loading} kontraSel={kontraSel}
                   onKontra={onKontra} toggleKontraUnit={toggleKontraUnit} />
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
