// Chrome + pure helpers for the Play tab: display maps, the card chip, the
// play-phase animation reducers and the speech-bubble hook. Everything here is
// STATE-FREE (or self-contained) — PlayVsAI.tsx keeps the game-flow state.
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

import type { PlayAnalysis, PlayState } from "./api";
import type { Card, Suit, Rank } from "./cards";
import { SUIT_HUN, SUIT_SYMBOL } from "./cards";

type Seat = 0 | 1 | 2;

export const SEAT_META: Record<Seat, { short: string; flavor: string; accent: string }> = {
  0: { short: "P0 · Forehand", flavor: "You act first in the auction.",
       accent: "linear-gradient(135deg, #6b1f8a 0%, #a23ed1 100%)" },
  1: { short: "P1 · Middle",   flavor: "Second to act.",
       accent: "linear-gradient(135deg, #2b5fa8 0%, #5089d6 100%)" },
  2: { short: "P2 · Rear",     flavor: "Third to act.",
       accent: "linear-gradient(135deg, #2b5fa8 0%, #5089d6 100%)" },
};

// Card order is decided ONCE, server-side, in ulti.card.sort_hand — it has to be, because
// the colourless contracts (betli / színtelen duri) read the Ten low and only the backend
// knows the contract. Every hand below is rendered in the order the API sent it; do not
// re-sort here or that rule silently stops applying.
export function placeholderHand(n: number, offset: number): Card[] {
  const out: Card[] = [];
  for (let i = 0; i < n; i++) out.push({ suit: "acorns", rank: "7", id: -(offset + i + 1) });
  return out;
}

export const PLAYER_LABEL: Record<number, string> = { 0: "P0", 1: "P1", 2: "P2" };
export const TRUMP_LABEL = (s: string | null | undefined): string =>
  s ? `${SUIT_SYMBOL[s as Suit]} ${SUIT_HUN[s as Suit]}` : "színtelen";
export const KONTRA_WORD: Record<number, string> = { 0: "", 1: "kontra", 2: "rekontra" };
// Play-index → role name (0 = soloist). Used by the play log and seat labels.
export const ROLE_LABEL: Record<number, string> = { 0: "Játékos", 1: "Védő 1", 2: "Védő 2" };
// Compact Hungarian rank labels for the play log.
export const RANK_SHORT: Record<Rank, string> = {
  "7": "7", "8": "8", "9": "9", lower: "alsó", upper: "felső", king: "K", "10": "10", ace: "ász",
};
// A move in the play log: the coloured suit icon (as on the trump buttons) + rank.
export function CardChip({ card }: { card: Card }) {
  return (
    <span className="play-log-card">
      <span className={`trump-symbol trump-${card.suit}`}>{SUIT_SYMBOL[card.suit]}</span>
      <span className="play-log-rank">{RANK_SHORT[card.rank]}</span>
    </span>
  );
}

// ── Play-phase animation (ported from BetliHu) ──────────────────────────────────
export const ANIM_STEP_MS = 850;
export const ANIM_TRICK_PAUSE_MS = 1250;

export function applyUserPlay(cur: PlayState, card: Card): PlayState {
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
export function playBaseline(resp: PlayState): PlayState {
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

export function applyStepToVisible(cur: PlayState, target: PlayState): PlayState {
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
export function useUltiBubble(): { show: (t: string) => void; clear: () => void; node: ReactNode } {
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


// One step of the analysis line (recorded play or explored branch).
export type EffectivePly = {
  ply_index: number; player_id: 0 | 1 | 2; chosen_card: Card; legal_card_ids: number[];
  verdict: PlayAnalysis["per_ply"][number] | null; by_ai: boolean; is_branch: boolean;
};
export interface AnalysisView {
  hands: Card[][];
  currentTrick: { player_id: 0 | 1 | 2; card: Card }[];
  activePlayer: 0 | 1 | 2 | null;
  legalIds: Set<number> | null;
  branchAtPly: number | null;
  currentPly: number;
  thisPly: EffectivePly | null;
}
