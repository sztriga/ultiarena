// Typed client for the apps/api FastAPI server.

import type { Card, Suit } from "./cards";
import { getAuth } from "./auth";

async function http<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {};
  if (body) headers["Content-Type"] = "application/json";
  const auth = getAuth();                      // logged in → every call carries identity
  if (auth) headers["Authorization"] = `Bearer ${auth.token}`;
  const resp = await fetch(`/api${path}`, {
    method,
    headers: Object.keys(headers).length ? headers : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${method} /api${path} → ${resp.status}: ${text}`);
  }
  return resp.json() as Promise<T>;
}

// ── Perfect-information solver probe ──────────────────────────────────────

export interface PisPVStep {
  player_id:      0 | 1 | 2;
  card:           Card;
  trick_index:    number;
  trick_position: number;
  legal_card_ids: number[];
}

export interface PisExploreRequest {
  hands:            number[][];          // 3 hands of card_ids
  soloist:          number;
  starting_leader:  number;
  total_tricks:     number;
  moves:            number[];             // card_ids played so far
  forced_card_id:   number;
  contract?:        string;               // SOLVE contract (betli|parti|durchmars|multi)
  trump?:           string | null;
  // Full-Ulti extras (needed to branch off a real play-tab deal):
  build_contract?:    string;             // BUILD contract (e.g. "parti" when solving "multi")
  talon?:             number[];
  declare_marriages?: boolean;
  marriage_restrict?: string | null;
  multi_weights?:     Record<string, number> | null;
}

export interface PisExploreResponse {
  alt_pv:        PisPVStep[];
  alt_start:     number;                  // = moves.length on the request
  value:         number;
  soloist_takes: number;
  verdict:       "soloist" | "defenders";
}

// ── Play (Ulti vs AI) ─────────────────────────────────────────────────────

export interface PlayLegalBid {
  kind:          "pass" | "start" | "bid";   // Passz · Kezdés (holder) · a contract
  rung_index:    number;         // -1 Passz · -2 Kezdés · else the ladder rung
  bid_index:     number;         // which interchangeable game on the rung
  label:         string;
  value:         number;
  piros:         boolean;
  colorless:     boolean;
  trump_options: string[];       // [] színtelen · ["hearts"] piros · 3 suits otherwise
}

export interface PlayAuctionCurrent {
  pid:        number;
  contract:   string;
  trump:      string | null;
  rung_index: number;
}

export interface PlayAuctionHistoryEntry {
  pid:         number;
  kind:        "bid" | "pass";
  contract?:   string;
  trump?:      string | null;
  rung_index?: number;
}

export interface PlayAuction {
  turn:          number | null;
  current:       PlayAuctionCurrent | null;
  passes:        number;
  history:       PlayAuctionHistoryEntry[];
  done:          boolean;
  winner:        number | null;
  is_human_turn: boolean;
  is_holder:     boolean;             // you own the standing bid → start play or raise
  reclaim:       boolean;             // forehand's last look after all-pass: bid or passz→pay
  awaiting_bid:  boolean;             // bid step: you hold 12 → dropdown + discard
  can_pickup:    boolean;             // auction step: you may Felveszem to bid
  opening:       boolean;             // true = no bid yet (you'd be opening)
  picked_up?:    boolean;             // in bid step because you took the talon up (ring it)
  legal_bids:    PlayLegalBid[] | null;
}

export interface PlayBubble { player: number; text: string; ply?: number; }

export interface PlayKontraUnit { key: string; label: string; }
export interface PlayKontra {
  pending:       { role: "def" | "sol"; play_index: number } | null;
  is_human_turn: boolean;
  role:          "def" | "sol";
  units:         PlayKontraUnit[];      // the units this decision can (re)kontra separately
  primary:       string | null;         // first available unit (back-compat display)
}

export interface PlayTrickPlay {
  player_id: 0 | 1 | 2;
  card:      Card;
}

export interface PlayHistoryStep {
  player_id:      0 | 1 | 2;
  card:           Card;
  trick_index:    number;
  trick_position: number;
  by_ai:          boolean;
}

export interface PlayResult {
  winner:         "soloist" | "defenders";
  made:           boolean;
  sol_gp_per_def: number;
  human_gp:       number;
  user_won:       boolean;
  contract:       string;
  kontra_level:   number;
  seat_gp:        number[];        // per real seat [P0, P1, P2], zero-sum
  soloist_seat?:  number;          // real seat of the soloist (opener/payer on all-pass)
  all_passed?:    boolean;         // true = obligatory pass penalty, no hand played
  silents?:       { key: string; label: string; gp: number }[];  // csendes contracts scored (soloist-perspective GP)
}

export interface PlayScore {
  sol_points:    number;             // soloist card points incl. declared marriages (full)
  def_points:    number;             // defender card points incl. marriages (+ talon if revealed)
  talon_points:  number | null;      // null = hidden from you (defender, mid-game)
  sol_card?:     number;             // soloist card points, marriage bonus removed
  def_card?:     number;             // defender card points, marriage bonus removed (+ talon)
  marr?:         number[];           // marriage declaration bonus per play-index [soloist, def1, def2]
  mode?:         "tricks" | "points";  // what the running tally means in this contract
  def_marriage_counts?: boolean;     // false in a bid 100 — defender marriages score nothing
  sol_tricks?:   number;             // tricks taken — the tally that matters in trick contracts
  def_tricks?:   number;
}

export interface PlayState {
  rev?: number;                       // server mutation counter — poll adoption key
  live?: boolean;                     // a lobby-table game (multiplayer)
  usernames?: Record<number, string>; // live: real seat → username ("Gép" for AI chairs)
  is_chooser?: boolean;               // trump_select: is it THIS viewer who declares?
  game_id: string;
  seat:    0 | 1 | 2;
  seed:    number;
  phase:   "bid" | "passed" | "trump_select" | "kontra" | "play" | "done";

  // trump_select phase (you won a plain colored game, declare the suit)
  trump_options?: string[];

  // bidding phase
  own_hand?:  Card[];
  bid_hand?:  Card[] | null;            // your 10 + the 2 talon cards, when it's your bid turn
  talon_ids?: number[];
  auction?:   PlayAuction;
  bubbles?:   PlayBubble[];             // speech-bubble events, keyed by play-index (drained once)

  // play phase (play-index space: 0 = soloist)
  soloist?:          0 | 1 | 2;
  human_play_index?: 0 | 1 | 2;
  contract?:         string;
  contract_value?:   number;
  trump?:            string | null;
  kontra_level?:     number;
  kontra?:           PlayKontra | null;
  score?:            PlayScore;
  captured?:         Card[];             // your own captured (won) cards
  talon_count?:      number;             // # talon cards (always 2) — rendered face-down
  talon?:            Card[] | null;      // the talon cards, revealed only at game end
  reveal_soloist?:   boolean;            // terített: soloist's hand is face-up (after trick 1)
  hands?:            (Card | null)[][];
  hand_sizes?:       number[];
  current_trick?:    PlayTrickPlay[];
  current_player?:   0 | 1 | 2 | null;
  legal_card_ids?:   number[] | null;
  history?:          PlayHistoryStep[];
  terminal?:         boolean;
  result?:           PlayResult | null;

  setup_ms?: number;
  step_ms?:  number;
}

export interface PlayAnalysisPly {
  ply_index:        number;
  player_id:        0 | 1 | 2;
  chosen_card:      Card;
  god_best_card:    Card;
  god_best_value:   number;
  god_chosen_value: number;
  is_blunder:       boolean;          // legacy, solver-unit; the UI shows `severity`
  legal_card_ids:   number[];
  by_ai:            boolean;
  // GP verdict. `gp_loss` is what the move cost in real game points, which is NOT the
  // solver's objective — that is a sum of binary indicators (parti ±1, ulti 0/1) in
  // which 81% of mid-game positions tie. `gp_loss_knowable` is the share of that cost
  // findable from the mover's own seat; a large loss with a small knowable share is bad
  // luck rather than a mistake. Null when the server ran with ANALYSIS_WORLDS=0.
  gp_seat_after?:    number[] | null;  // value AFTER the move, one per PLAY index
  gp_chosen?:        number | null;
  gp_best?:          number | null;
  gp_loss?:          number | null;
  gp_swing?:         number | null;
  gp_loss_knowable?: number | null;
  gp_best_card?:     Card | null;
  severity?:         "ok" | "pontatlanság" | "hiba" | "baklövés" | null;
}

export interface PlayAnalysis {
  game_id:           string;
  contract:          string;             // display label of the game
  solve_contract:    string;             // betli | multi | durchmars (for /pis/explore)
  build_contract:    string;
  marriage_restrict: string | null;
  multi_weights:     Record<string, number> | null;
  declare_marriages: boolean;
  soloist:           0 | 1 | 2;          // always 0 in play-index space
  human_play_index:  0 | 1 | 2;
  leader:            0 | 1 | 2;
  trump:             string | null;
  initial_hands:     Card[][];           // play-index space (soloist=0)
  talon:             Card[];
  per_ply:           PlayAnalysisPly[];
  analysis_ms:       number;
}

export interface PlayNewRequest { seat: 0 | 1 | 2; seed?: number; device_id?: string; }
export interface PlayBidRequest {
  game_id: string; rung_index: number; bid_index: number; trump: string | null; discard_ids: number[];
}

// ── Accounts + the live lobby (docs/MULTIPLAYER.md) ─────────────────────────

export interface LiveSeat {
  username: string; user_id: string; is_host: boolean; is_me: boolean;
}
export interface LiveTable {
  table_id: string;
  host: string;
  is_host: boolean;
  seats: (LiveSeat | null)[];
  full: boolean;
  invited_me: boolean;
  state: string;
  game_id: string | null;
}
export interface LiveChatMsg { seq: number; user: string; text: string; ts: number; }
export interface LivePoll {
  me: string;
  members: string[];
  chat: LiveChatMsg[];
  tables: LiveTable[];
  my_table: string | null;
}

// ── Profile (/me/*) ─────────────────────────────────────────────────────────

export interface MeGame {
  id: string; created_at: number; seed: number; contract: string;
  trump: string | null; soloist_seat: number; kontra_level: number;
  winner: string; made: boolean;
  my_seat: number; i_was_soloist: boolean; my_gp: number;
}
export interface MeContractStat {
  contract: string; n: number; gp: number; gp_mean: number;
  sol_n: number; sol_made: number;
}
export interface MeStats {
  games: number; gp_total?: number; gp_per_game?: number; wins?: number;
  as_soloist?: { n: number; made: number; gp: number };
  contracts?: MeContractStat[];
}

/** One live game of this browser, as listed on the splash (resume). */
export interface PlayOngoing {
  game_id: string;
  phase: string;                 // bid | trump_select | kontra | play | done
  seat: number;
  contract: string | null;       // null until the auction resolves
  trump: string | null;
  idle_s: number;
}

// ── Villámtalon puzzle ──────────────────────────────────────────────────────

export interface PuzzlePuzzle {
  puzzle_id: string;
  contract: string;
  contract_label: string;
  trump: Suit | null;
  prompt: string;
  difficulty: number;
  cards: Card[];
}
export interface PuzzleResult {
  points: number;
  quality: number;
  your_p: number;
  best_p: number;
  optimal_ids: number[];
  your_ids: number[];
  perfect: boolean;
  combo: number;
  counted: boolean;
}
export interface PuzzleState {
  game_id: string;
  score: number;
  combo: number;
  best_combo: number;
  solved: number;
  time_left: number;
  duration: number;
  done: boolean;
  last_result: PuzzleResult | null;
  puzzle: PuzzlePuzzle | null;
}

export const api = {
  health:   () => http<{ status: string }>("GET", "/health"),
  pisExplore:      (req: PisExploreRequest)      => http<PisExploreResponse>("POST", "/pis/explore", req),
  playNew:    (req: PlayNewRequest)                        => http<PlayState>("POST", "/play/new", req),
  playBid:    (req: PlayBidRequest)                        => http<PlayState>("POST", "/play/bid", req),
  playPickup: (game_id: string)                            => http<PlayState>("POST", "/play/pickup", { game_id }),
  playPass:   (game_id: string, discard_ids: number[] = []) => http<PlayState>("POST", "/play/pass", { game_id, discard_ids }),
  playTrump:  (game_id: string, trump: string)             => http<PlayState>("POST", "/play/trump", { game_id, trump }),
  playKontra: (req: { game_id: string; units: string[] }) => http<PlayState>("POST", "/play/kontra", req),
  playAnalysis: (req: { game_id: string }) => http<PlayAnalysis>("POST", "/play/analysis", req),
  playMove:   (req: { game_id: string; card_id: number }) => http<PlayState>("POST", "/play/move", req),
  playState:  (game_id: string)                            => http<PlayState>("POST", "/play/state", { game_id }),
  playMine:   (device_id: string)                          => http<{ games: PlayOngoing[] }>("POST", "/play/mine", { device_id }),
  playDelete: (game_id: string, device_id: string)         => http<{ deleted: boolean }>("DELETE", `/play/session/${game_id}?device=${device_id}`),
  authRegister: (req: { username: string; password: string; device_id?: string }) =>
    http<{ token: string; username: string }>("POST", "/auth/register", req),
  authLogin:    (req: { username: string; password: string; device_id?: string }) =>
    http<{ token: string; username: string }>("POST", "/auth/login", req),
  authLogout:   () => http<{ ok: boolean }>("POST", "/auth/logout", {}),
  authDevLogin: () => http<{ token: string; username: string }>("POST", "/auth/devlogin", {}),
  meGames: (device: string, cursor?: number | null, limit = 15) =>
    http<{ games: MeGame[]; next_cursor: number | null }>(
      "GET", `/me/games?device=${device}&limit=${limit}${cursor ? `&cursor=${cursor}` : ""}`),
  meStats:    (device: string) => http<MeStats>("GET", `/me/stats?device=${device}`),
  meNickname: (username: string) => http<{ username: string }>("POST", "/me/nickname", { username }),
  meAnalysis: (id: string, device: string) =>
    http<PlayAnalysis>("POST", `/me/games/${id}/analysis?device=${device}`, {}),
  livePoll:     (chat_after: number) => http<LivePoll>("POST", "/live/poll", { chat_after }),
  liveChat:     (text: string)       => http<{ ok: boolean }>("POST", "/live/chat", { text }),
  tableCreate:  ()                   => http<{ table_id: string }>("POST", "/live/table/create", {}),
  tableJoin:    (table_id: string)   => http<{ table_id: string; seat: number }>("POST", "/live/table/join", { table_id }),
  tableLeave:   (table_id: string)   => http<{ ok: boolean }>("POST", "/live/table/leave", { table_id }),
  tableKick:    (table_id: string, user_id: string) => http<{ ok: boolean }>("POST", "/live/table/kick", { table_id, user_id }),
  tableInvite:  (table_id: string, username: string) => http<{ ok: boolean }>("POST", "/live/table/invite", { table_id, username }),
  tableStart:   (table_id: string)   => http<{ game_id: string }>("POST", "/live/table/start", { table_id }),
  puzzleNew:   ()                                            => http<PuzzleState>("POST", "/puzzle/new"),
  puzzleSolve: (req: { game_id: string; discard_ids: number[] }) => http<PuzzleState>("POST", "/puzzle/solve", req),
  puzzleState: (game_id: string)                            => http<PuzzleState>("GET",  `/puzzle/state/${game_id}`),
  puzzleEnd:   (game_id: string)                            => http<PuzzleState>("POST", `/puzzle/end/${game_id}`),
};

