// Card render — uses the JPG images from /cards (mounted by the FastAPI server).

import type { Card } from "./cards";
import { cardBackUrl, cardLabel, cardUrl, isPointCard } from "./cards";

export type CardSize = "micro" | "thumb" | "hand" | "trick";

const SIZE_CLASS: Record<CardSize, string> = {
  micro: "card-micro",
  thumb: "card-thumb",
  hand:  "card-hand",
  trick: "card-trick",
};

export interface CardViewProps {
  card: Card;
  size?: CardSize;
  highlighted?: boolean;
  dimmed?: boolean;
  playable?: boolean;
  /** Chosen for discard — lifts the card up. */
  selected?: boolean;
  /** Marked (e.g. a picked-up talon card) — a subtle ring. */
  marked?: boolean;
  onClick?: (card: Card) => void;
}

export function CardView({
  card,
  size = "hand",
  highlighted = false,
  dimmed = false,
  playable = false,
  selected = false,
  marked = false,
  onClick,
}: CardViewProps) {
  const cls = [
    SIZE_CLASS[size],
    highlighted ? "card-highlight" : "",
    dimmed ? "card-dim" : "",
    playable ? "card-playable" : "",
    selected ? "card-selected" : "",
    marked ? "card-marked" : "",
    isPointCard(card) ? "card-points" : "",
    onClick ? "card-clickable" : "",
  ]
    .filter(Boolean)
    .join(" ");
  const img = (
    <img
      className={cls}
      src={cardUrl(card)}
      alt={cardLabel(card)}
      title={cardLabel(card)}
      onClick={onClick ? () => onClick(card) : undefined}
      draggable={false}
    />
  );
  // `marked` is only ever a picked-up talon card in this app — tag it "talon".
  if (marked) {
    return (
      <span className="card-mark-wrap">
        {img}
        <span className="card-mark-badge">talon</span>
      </span>
    );
  }
  return img;
}

export interface CardBackProps {
  size?: CardSize;
}

export function CardBack({ size = "hand" }: CardBackProps) {
  return <img className={SIZE_CLASS[size]} src={cardBackUrl()} alt="card back" draggable={false} />;
}

export interface HandViewProps {
  cards: Card[];
  size?: CardSize;
  highlightId?: number;
  dimIds?: number[];
  /** Card ids that should pop up as the currently-playable choices. */
  playableIds?: number[];
  /** Card ids chosen for discard (lifted). */
  selectedIds?: Set<number>;
  /** Card ids marked (e.g. picked-up talon) — a subtle ring. */
  markIds?: Set<number>;
  onCardClick?: (card: Card) => void;
  /** Overlap cards into a fan instead of spreading them out. */
  fan?: boolean;
}

export function HandView({
  cards,
  size = "hand",
  highlightId,
  dimIds = [],
  playableIds = [],
  selectedIds,
  markIds,
  onCardClick,
  fan = false,
}: HandViewProps) {
  return (
    <div className={fan ? "hand-row hand-fan" : "hand-row"}>
      {cards.map((c) => (
        <CardView
          key={c.id}
          card={c}
          size={size}
          highlighted={c.id === highlightId}
          dimmed={dimIds.includes(c.id)}
          playable={playableIds.includes(c.id)}
          selected={selectedIds?.has(c.id)}
          marked={markIds?.has(c.id)}
          onClick={onCardClick}
        />
      ))}
    </div>
  );
}
