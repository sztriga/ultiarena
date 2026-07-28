// Card types and image-URL helpers for the Hungarian (Tell-pattern) deck.
// Image filenames follow trickster's convention so we can reuse those assets:
//   /cards/card_piece_<S><R>.jpg     S ∈ {A, L, H, B}
//                                    R ∈ {7, 8, 9, J, Q, K, 10, A}
// where J = lower (alsó) and Q = upper (felső).

export type Suit = "acorns" | "leaves" | "hearts" | "bells";
export type Rank = "7" | "8" | "9" | "lower" | "upper" | "king" | "10" | "ace";

export interface Card {
  suit: Suit;
  rank: Rank;
  id: number;
}

export const SUITS: Suit[] = ["acorns", "leaves", "hearts", "bells"];
export const RANKS: Rank[] = ["7", "8", "9", "lower", "upper", "king", "10", "ace"];

const SUIT_CHAR: Record<Suit, string> = {
  acorns: "A",
  leaves: "L",
  hearts: "H",
  bells:  "B",
};

const RANK_CHAR: Record<Rank, string> = {
  "7":   "7",
  "8":   "8",
  "9":   "9",
  lower: "J",   // alsó
  upper: "Q",   // felső
  king:  "K",
  "10":  "10",
  ace:   "A",
};

// Hungarian rank/suit labels for tooltips and the friend's terminology.
export const RANK_HUN: Record<Rank, string> = {
  "7": "hetes",
  "8": "nyolcas",
  "9": "kilences",
  lower: "alsó",
  upper: "felső",
  king: "király",
  "10": "tízes",
  ace: "ász",
};

export const SUIT_HUN: Record<Suit, string> = {
  acorns: "makk",
  leaves: "zöld",
  hearts: "piros",
  bells:  "tök",
};

// Suit symbols used in trump badges.
export const SUIT_SYMBOL: Record<Suit, string> = {
  hearts: "♥",
  bells:  "♣",
  leaves: "♠",
  acorns: "♦",
};

export function cardUrl(card: Card): string {
  return `/cards/card_piece_${SUIT_CHAR[card.suit]}${RANK_CHAR[card.rank]}.jpg`;
}

export function cardBackUrl(): string {
  return "/cards/card_back.png";
}

export function cardLabel(card: Card): string {
  return `${RANK_HUN[card.rank]} ${SUIT_HUN[card.suit]}`;
}

export function isPointCard(card: Card): boolean {
  return card.rank === "10" || card.rank === "ace";
}

export function rankPower(card: Card): number {
  return RANKS.indexOf(card.rank);
}
