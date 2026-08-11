// id → (suit, rank) in the backend's encoding (suit_index*8 + rank_index).
import type { Suit, Rank } from "../ui/cards";
export const SUITS: Suit[] = ["hearts", "bells", "leaves", "acorns"];
export const RANKS: Rank[] = ["7", "8", "9", "lower", "upper", "king", "10", "ace"];
