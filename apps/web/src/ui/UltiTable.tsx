// Shared 3-zone Ulti table: defenders on top, trick in the middle, soloist /
// declarer fanned hand at the bottom. Used by both the Spectator (full game)
// and the PIS tab (open-handed Betli probe).
//
// The table itself is purely presentational — branching logic and step state
// live in the consumer. Cards in the active player's hand whose ids aren't in
// `legalIds` get the dim treatment and don't fire `onCardClick`.

import type { ReactNode } from "react";

import { CardBack, CardView, HandView } from "./CardView";
import { cardLabel, cardUrl, type Card } from "./cards";

export interface SeatChrome {
  /** Inline label content (name, role tag, etc.). */
  label: ReactNode;
  /** Optional pre-rendered (timed) speech-bubble node shown over the seat. */
  bubbleNode?: ReactNode;
}

export interface TrickPlay {
  player_id: 0 | 1 | 2;
  card:      Card;
}

export interface UltiTableProps {
  /** Already-sorted hands for each seat. */
  hands:        [Card[], Card[], Card[]];
  /** Per-seat label + bubble content. */
  seats:        { 0: SeatChrome; 1: SeatChrome; 2: SeatChrome };
  /** Cards currently visible in the trick. */
  currentTrick: TrickPlay[];

  /** Whose turn it is (or null when the table is showing a non-decision step). */
  activePlayer: 0 | 1 | 2 | null;
  /** Legal card ids for the active player; `null` disables dimming. */
  legalIds:     Set<number> | null;
  /** Click handler. Cards not in `legalIds` swallow the click. */
  onCardClick?: (card: Card) => void;

  /** Optional content above the defender row (contract badge, alt marker, …). */
  aboveDefenders?: ReactNode;
  /** Optional content between the trick area and the soloist hand
   *  (talon strip, status bar, …). */
  belowTrick?:     ReactNode;

  /** Seat names — defaults supplied by consumer if needed. */
  seatNames: { 0: string; 1: string; 2: string };

  /** Seats whose cards should render as backs (hidden). Use card-count from
   *  `hands[seat].length`. */
  hiddenSeats?: Set<0 | 1 | 2>;

  /** Which seat should appear at the bottom (fanned, large) — default 0
   *  (classic ulti soloist-at-bottom). Set this to the *viewer's* seat in
   *  playable views so the user is always at the front. */
  bottomSeat?: 0 | 1 | 2;

  /** Bottom-hand cards chosen for discard (lifted) — used in the bid phase. */
  selectedIds?: Set<number>;
  /** Bottom-hand cards marked (e.g. picked-up talon) — a subtle ring. */
  markIds?: Set<number>;
  /** Hide the central trick "playing area" (e.g. during bidding). */
  hideTrick?: boolean;
}

function SeatHand({
  cards,
  chrome,
  isActive,
  legalIds,
  onCardClick,
  hidden = false,
}: {
  cards:       Card[];
  chrome:      SeatChrome;
  isActive:    boolean;
  legalIds:    Set<number> | null;
  onCardClick: ((c: Card) => void) | undefined;
  hidden?:     boolean;
}) {
  const click = isActive ? onCardClick : undefined;
  return (
    <div className={`ulti-ai-hand ${isActive ? "is-active" : ""}`}>
      <div className="ulti-ai-label">{chrome.label}</div>
      <div className="ulti-card-row">
        {hidden
          ? cards.map((_, i) => <CardBack key={i} size="thumb" />)
          : cards.map((c) => {
              const illegal = isActive && legalIds !== null && !legalIds.has(c.id);
              return (
                <CardView
                  key={c.id}
                  card={c}
                  size="thumb"
                  dimmed={illegal}
                  onClick={illegal ? undefined : click}
                />
              );
            })}
      </div>
      <div className="ulti-bubble-anchor">
        {chrome.bubbleNode}
      </div>
    </div>
  );
}

function SoloistHand({
  cards,
  chrome,
  isActive,
  legalIds,
  selectedIds,
  markIds,
  onCardClick,
  hidden = false,
}: {
  cards:       Card[];
  chrome:      SeatChrome;
  isActive:    boolean;
  legalIds:    Set<number> | null;
  selectedIds?: Set<number>;
  markIds?:     Set<number>;
  onCardClick: ((c: Card) => void) | undefined;
  hidden?:     boolean;
}) {
  const click = isActive ? onCardClick : undefined;
  const dimIds =
    isActive && legalIds !== null
      ? cards.filter((c) => !legalIds.has(c.id)).map((c) => c.id)
      : [];
  const playableIds =
    isActive && legalIds !== null
      ? cards.filter((c) => legalIds.has(c.id)).map((c) => c.id)
      : [];
  const guarded = click
    ? (c: Card) => {
        if (legalIds === null || legalIds.has(c.id)) click(c);
      }
    : undefined;
  return (
    <div className="ulti-bottom-seat">
      <div className="ulti-bubble-anchor" style={{ height: 0, position: "relative" }}>
        {chrome.bubbleNode}
      </div>
      <div className={`ulti-player-role ${isActive ? "" : "muted"}`}>
        {chrome.label}
      </div>
      {hidden ? (
        <div className="hand-row hand-fan">
          {cards.map((_, i) => <CardBack key={i} size="hand" />)}
        </div>
      ) : (
        <HandView
          cards={cards}
          size="hand"
          dimIds={dimIds}
          playableIds={playableIds}
          selectedIds={selectedIds}
          markIds={markIds}
          fan
          onCardClick={guarded}
        />
      )}
    </div>
  );
}

function TrickArea({
  currentTrick,
  seatNames,
  bottomSeat = 0,
}: {
  currentTrick: TrickPlay[];
  seatNames:    { 0: string; 1: string; 2: string };
  bottomSeat?:  0 | 1 | 2;
}) {
  // Slot order: [left-top-opponent, bottom-self, right-top-opponent].
  // Matches the seat rotation in the main UltiTable layout.
  // Counterclockwise: next player sits on the right.
  const top1: 0 | 1 | 2 = ((bottomSeat + 2) % 3) as 0 | 1 | 2;  // left
  const top2: 0 | 1 | 2 = ((bottomSeat + 1) % 3) as 0 | 1 | 2;  // right
  const order: Array<0 | 1 | 2> = [top1, bottomSeat, top2];
  const slots: Record<number, TrickPlay | undefined> = {};
  for (const play of currentTrick) slots[play.player_id] = play;
  return (
    <div className="ulti-trick-area">
      {order.map((pid) => {
        const play = slots[pid];
        return (
          <div key={pid} className="ulti-trick-slot">
            <div className="ulti-trick-player">
              {seatNames[pid]}
            </div>
            {play
              ? <CardView card={play.card} size="trick" />
              : <div className="ulti-trick-empty" />}
          </div>
        );
      })}
    </div>
  );
}

/** Inline talon strip — useful as `aboveDefenders` (PIS) or `belowTrick` (Spectator). */
export function TalonStrip({ talon }: { talon: Card[] }) {
  if (!talon.length) return null;
  return (
    <div className="row" style={{ justifyContent: "center" }}>
      <div className="ulti-talon-inline">
        <span>Talon:</span>
        {talon.map((c) => (
          <img
            key={c.id}
            className="ulti-talon-card"
            src={cardUrl(c)}
            alt={cardLabel(c)}
            title={cardLabel(c)}
          />
        ))}
      </div>
    </div>
  );
}

export function UltiTable(props: UltiTableProps) {
  const {
    hands,
    seats,
    currentTrick,
    activePlayer,
    legalIds,
    onCardClick,
    aboveDefenders,
    belowTrick,
    seatNames,
    hiddenSeats,
    bottomSeat = 0,
    selectedIds,
    markIds,
    hideTrick = false,
  } = props;
  const isHidden = (s: 0 | 1 | 2) => hiddenSeats?.has(s) ?? false;

  // Rotate which seat sits at the bottom (fanned), while keeping all data
  // keyed by the *real* player id everywhere else. Play order is
  // counterclockwise: the *next* player (bottomSeat + 1) sits on your RIGHT,
  // the one after them (bottomSeat + 2) sits on your LEFT.
  const top1: 0 | 1 | 2 = ((bottomSeat + 2) % 3) as 0 | 1 | 2;  // left
  const top2: 0 | 1 | 2 = ((bottomSeat + 1) % 3) as 0 | 1 | 2;  // right

  return (
    <div className="table">
      {aboveDefenders}
      <div className="ulti-ai-row">
        <SeatHand
          cards={hands[top1]}
          chrome={seats[top1]}
          isActive={activePlayer === top1}
          legalIds={activePlayer === top1 ? legalIds : null}
          onCardClick={onCardClick}
          hidden={isHidden(top1)}
        />
        <SeatHand
          cards={hands[top2]}
          chrome={seats[top2]}
          isActive={activePlayer === top2}
          legalIds={activePlayer === top2 ? legalIds : null}
          onCardClick={onCardClick}
          hidden={isHidden(top2)}
        />
      </div>
      {!hideTrick && (
        <TrickArea
          currentTrick={currentTrick}
          seatNames={seatNames}
          bottomSeat={bottomSeat}
        />
      )}
      {belowTrick}
      <SoloistHand
        cards={hands[bottomSeat]}
        chrome={seats[bottomSeat]}
        isActive={activePlayer === bottomSeat}
        legalIds={activePlayer === bottomSeat ? legalIds : null}
        selectedIds={selectedIds}
        markIds={markIds}
        onCardClick={onCardClick}
        hidden={isHidden(bottomSeat)}
      />
    </div>
  );
}
