// Typed client for the apps/api FastAPI server.

import type { Card, Suit } from "./cards";

export interface AgentInfo {
  id: string;
  label: string;
  kind: "heuristic" | "sb3";
  description: string;
}

export interface UberReason {
  kind: "lead" | "must_uber" | "follow_suit" | "must_trump" | "free";
  led_suit: string | null;
  beat_rank: string | null;
}

export interface TrumpChoice {
  action: number;
  suit: Suit;
  family: "parti" | "40-100" | "ulti" | "40-100+ulti";
  is_piros: boolean;
}

export interface Move {
  step: number;
  phase: "TRUMP" | "DISCARD" | "PLAY";
  player_id: 0 | 1 | 2;
  actor: string;
  legal_actions: number[];
  action: number;
  card: Card | null;
  trump_choice: TrumpChoice | null;
  uber: UberReason | null;
  trick_index: number | null;
  trick_position: number | null;
  hand_before: Card[];
  licit_after: string | null;
  payoffs_after: number[];
  trick_scores: number[];
  payoff_delta: number[];
}

export interface TrickPlay {
  player_id: 0 | 1 | 2;
  actor: string;
  card: Card;
}

export interface TrickSummary {
  index: number;
  leader: 0 | 1 | 2;
  plays: TrickPlay[];
  winner: 0 | 1 | 2;
  points: number;
  trick_scores: number[];
  running_payoffs: number[];
}

export interface GameResult {
  trump_suit: Suit | null;
  licit: string | null;
  talon: Card[];
  talon_points: number;
  trick_scores: number[];
  king_upper_points: number[];
  payoffs: number[];
  game_points: number[];
  declarer_wins: boolean;
  declarer_ulti: boolean;
  defender_ulti: boolean;
  declarer_has_trump_pair: boolean;
  defender_has_trump_pair: boolean;
}

export interface SpectateRequest {
  declarer: string;
  defender1: string;
  defender2: string;
  seed?: number;
  agent_seed?: number;
}

export interface UltiSpectateRequest extends SpectateRequest {
  alpha?:      number;
  suit_sigma?: number;
}

export interface SpectateResponseConfig {
  declarer:   string;
  defender1:  string;
  defender2:  string;
  // Server materializes these; always present on the response even if the
  // request omitted them.
  seed:       number;
  agent_seed: number;
  // Ulti-mode only — present when the response came from /spectate/ulti.
  alpha?:      number;
  suit_sigma?: number;
  trump?:      Suit;
}

export interface SpectateResponse {
  config: SpectateResponseConfig;
  initial_hands: Card[][];
  moves: Move[];
  tricks: TrickSummary[];
  result: GameResult;
}

export interface SpectateExploreRequest extends SpectateRequest {
  moves: number[];        // action ids already played
  forced_action: number;  // action to play instead of the original next move
}

export interface UltiSpectateExploreRequest extends UltiSpectateRequest {
  moves: number[];
  forced_action: number;
}

export interface SpectateExploreResponse {
  alt_start: number;
  alt_moves: Move[];
  alt_tricks: TrickSummary[];
  alt_first_trick_index: number;
  alt_result: GameResult;
}

async function http<T>(method: string, path: string, body?: unknown): Promise<T> {
  const resp = await fetch(`/api${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${method} /api${path} → ${resp.status}: ${text}`);
  }
  return resp.json() as Promise<T>;
}

// ── Perfect-information solver probe ──────────────────────────────────────

export interface PisProbeRequest {
  seed?:      number;
  alpha?:     number;
  hand_size?: number;        // legacy; backend always solves the full 10-card game
}

export interface PisPVStep {
  player_id:      0 | 1 | 2;
  card:           Card;
  trick_index:    number;
  trick_position: number;
  legal_card_ids: number[];
}

export interface PisDiagnostics {
  nodes:              number;
  terminals:          number;             // currently zeroed by the Cython solver
  cutoffs:            number;
  early_terminations: number;             // currently zeroed
  max_depth:          number;             // currently zeroed
  cutoff_pct:         number;
  tt_hits:            number;
  tt_stores:          number;
  tt_hit_pct:         number;
  endgame_cards:      number;
  endgame_plies:      number;
  best_ms:            number;
  root_ms:            number;
  pv_ms:              number;
  total_ms:           number;
  pv_plies:           number;
}

export interface PisProbeResponse {
  config: {
    seed:      number | null;
    alpha:     number;
    hand_size: number;
    contract:  string;
  };
  hands:                  Card[][];
  talon:                  Card[];
  soloist:                0 | 1 | 2;
  leader:                 0 | 1 | 2;
  verdict:                "soloist" | "defenders";
  value:                  number;
  soloist_takes:          number;
  best_lead:              Card | null;
  all_leads:              { card: Card; value: number }[];
  principal_variation:    PisPVStep[];
  solve_ms:               number;
  diagnostics:            PisDiagnostics;
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

// ── PIMC play-through (god annotates every move) ──────────────────────────

export interface PimcPlayRequest {
  seed?:      number;
  alpha?:     number;
  n_samples?: number;
}

export interface PimcStep extends PisPVStep {
  god_best_value:    number;       // best value any legal move could have achieved
  god_chosen_value:  number;       // value of the move PIMC actually chose
  is_blunder:        boolean;
  pimc_avg:          Record<number, number>;   // card_id → averaged sample value
}

export interface PimcPlayResponse {
  config: {
    seed:      number | null;
    alpha:     number;
    n_samples: number;
    contract:  string;
  };
  hands:         Card[][];
  talon:         Card[];
  soloist:       0 | 1 | 2;
  leader:        0 | 1 | 2;
  play_log:      PimcStep[];
  verdict:       "soloist" | "defenders";
  soloist_takes: number;
  diagnostics: {
    total_ms:     number;
    n_decisions:  number;
    n_blunders:   number;
    blunder_rate: number;
  };
}

// ── Betli.hu (playable game) ────────────────────────────────────────────────

export type BetliHuDifficulty = "easy" | "medium" | "hard" | "milcsi";
export type BetliHuContract   = "betli" | "parti";

export interface BetliHuNewRequest {
  contract?:  BetliHuContract;        // defaults to "betli" server-side
  seat:       0 | 1 | 2;
  difficulty: BetliHuDifficulty;
  seed?:      number;
}

export interface BetliHuMoveRequest {
  game_id: string;
  card_id: number;
}

export interface BetliHuStateRequest {
  game_id: string;
}

export interface BetliHuHistoryStep {
  player_id:      0 | 1 | 2;
  card:           Card;
  trick_index:    number;
  trick_position: number;
  by_ai:          boolean;
}

export interface BetliHuTrickPlay {
  player_id: 0 | 1 | 2;
  card:      Card;
}

export interface BetliHuState {
  game_id:        string;
  contract:       BetliHuContract;
  seat:           0 | 1 | 2;
  difficulty:     BetliHuDifficulty;
  alpha:          number;
  seed:           number;
  soloist:        0 | 1 | 2;
  leader:         0 | 1 | 2;
  trump:          string | null;
  sol_points:     number | null;     // parti only — soloist's accumulated card-points
  talon:          Card[];
  hands:          (Card | null)[][];   // hidden seats are arrays of null (length = #cards)
  hand_sizes:     number[];
  current_trick:  BetliHuTrickPlay[];
  current_player: 0 | 1 | 2 | null;    // null if terminal
  legal_card_ids: number[] | null;     // present only when it's the user's turn
  history:        BetliHuHistoryStep[];
  terminal:       boolean;
  winner:         "soloist" | "defenders" | null;
  user_won:       boolean | null;
  setup_ms?:      number;
  move_ms?:       number;
}

export interface BetliHuAnalysisPly {
  ply_index:        number;
  player_id:        0 | 1 | 2;
  chosen_card:      Card;
  god_best_card:    Card;
  god_best_value:   number;
  god_chosen_value: number;
  is_blunder:       boolean;
  legal_card_ids:   number[];
  by_ai:            boolean;
}

export interface BetliHuAnalysis {
  game_id:       string;
  contract:      BetliHuContract;
  seat:          0 | 1 | 2;
  soloist:       0 | 1 | 2;
  leader:        0 | 1 | 2;
  trump:         string | null;
  initial_hands: Card[][];
  talon:         Card[];
  per_ply:       BetliHuAnalysisPly[];
  analysis_ms:   number;
}

// ── Play (full-ladder Ulti vs the frontier champion) ────────────────────────

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
}

export interface PlayState {
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
  is_blunder:       boolean;
  legal_card_ids:   number[];
  by_ai:            boolean;
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

export interface PlayNewRequest { seat: 0 | 1 | 2; seed?: number; }
export interface PlayBidRequest {
  game_id: string; rung_index: number; bid_index: number; trump: string | null; discard_ids: number[];
}

// ── Training jobs ───────────────────────────────────────────────────────────

export interface TrainTierField {
  name:    string;
  type:    "int" | "float" | "str" | string;
  default: number | string;
  help?:   string;
}

export interface TrainTierPreset {
  id:          string;
  label:       string;
  description: string;
  values:      Record<string, number | string>;
}

export interface TrainPresetsResponse {
  presets:    TrainTierPreset[];
  fields:     TrainTierField[];
  ppo_help?:  Record<string, string>;
  ppo_stages?: string[];
}

export interface StartUltiTrainRequest {
  tier:         string;
  save_id:      string;
  overrides?:   Record<string, number | string>;
  register_as?: string;
  seed?:        number;
}

export interface TrainProgressPoint {
  step: number; games: number; winrate: number;
  vloss: number; ploss: number; lr: number; elapsed_s: number;
}

export interface TrainJobState {
  id:            string;
  kind:          "ulti_az" | "oldtawer_ppo";
  label:         string;
  save_dir:      string;
  config:        Record<string, unknown>;
  register_as:   string | null;
  seed:          number;
  status:        "pending" | "running" | "done" | "failed";
  started_at:    number | null;
  ended_at:      number | null;
  error:         string | null;
  elapsed_s:     number;

  // ulti_az fields
  cur_step:      number;
  total_steps:   number;
  games:         number;
  winrate:       number;
  vloss:         number;
  ploss:         number;
  history:       TrainProgressPoint[];

  // oldtawer_ppo fields
  log_tail:         string[];
  current_stage:    string | null;
  completed_stages: string[];
  total_stages:     number;

  registered_as: string | null;
}

export interface StartOldtawerTrainRequest {
  run_id:                   string;
  from_stage?:              string;
  n_envs?:                  number;
  seed?:                    number;
  patience?:                number;
  parti_steps?:             number;
  parti_pregame_steps?:     number;
  composite_turns?:         number;
  composite_play_steps?:    number;
  composite_pregame_steps?: number;
  defender_turns?:          number;
  defender_steps_first?:    number;
  defender_steps?:          number;
  declarer_finetune_steps?:  number;
  pregame_recal_steps?:     number;
  orch_steps?:              number;
  orch_patience?:           number;
  register_as?:             string;
}

export interface TensorboardStatus {
  url:     string;
  running: boolean;
}

// ── Arena ───────────────────────────────────────────────────────────────────

export interface ArenaRequest {
  soloist:    string;
  defender1:  string;
  defender2:  string;
  n_games?:   number;
  seed?:      number;
  agent_seed?: number;
  alpha?:     number;
  suit_sigma?: number;
}

export interface ArenaPerSoloist {
  games:         number;
  wins:          number;
  win_rate:      number;
  avg_payoff:    number;
  avg_gp:        number;
  avg_tricks:    number;
  // Sums kept on the response — useful for drill-down.
  payoff_sum?:   number;
  gp_sum?:       number;
  tricks_sum?:   number;
}

export interface ArenaGameLog {
  seed:           number;
  soloist:        string;
  def1:           string;
  def2:           string;
  declarer_wins:  boolean;
  payoffs:        number[];
  game_points:    number[];
  soloist_tricks: number;
}

export interface ArenaResponse {
  config: {
    agents:     string[];
    n_games:    number;
    seed:       number;
    alpha:      number;
    suit_sigma: number;
  };
  totals: {
    games:          number;
    declarer_wins:  number;
  };
  per_soloist: Record<string, ArenaPerSoloist>;
  games:       ArenaGameLog[];
  elapsed_s:   number;
}

// ── Villámtalon (puzzle rush) ──────────────────────────────────────────────
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
  agents:   () => http<{ agents: AgentInfo[] }>("GET", "/agents"),
  spectate:        (req: SpectateRequest)        => http<SpectateResponse>("POST", "/spectate", req),
  spectateUlti:        (req: UltiSpectateRequest)        => http<SpectateResponse>("POST", "/spectate/ulti", req),
  spectateUltiExplore: (req: UltiSpectateExploreRequest) => http<SpectateExploreResponse>("POST", "/spectate/ulti/explore", req),
  spectateExplore:     (req: SpectateExploreRequest)     => http<SpectateExploreResponse>("POST", "/spectate/explore", req),
  pis:             (req: PisProbeRequest)        => http<PisProbeResponse>("POST", "/pis/probe", req),
  pisExplore:      (req: PisExploreRequest)      => http<PisExploreResponse>("POST", "/pis/explore", req),
  pisPimcPlay:     (req: PimcPlayRequest)        => http<PimcPlayResponse>("POST", "/pis/pimc_play", req),
  trainPresets:    () => http<TrainPresetsResponse>("GET", "/train/presets"),
  startUltiTrain:  (req: StartUltiTrainRequest) => http<{ job_id: string; status: string; save_dir: string }>("POST", "/train/ulti", req),
  startOldtawerTrain: (req: StartOldtawerTrainRequest) => http<{ job_id: string; status: string; save_dir: string }>("POST", "/train/oldtawer", req),
  trainJobs:       () => http<{ jobs: TrainJobState[] }>("GET", "/train/jobs"),
  trainJob:        (id: string) => http<TrainJobState>("GET", `/train/jobs/${id}`),
  killTrainJob:    (id: string) =>
    http<{ killed: boolean; job_id?: string; status?: string; reason?: string }>(
      "POST", `/train/jobs/${id}/kill`,
    ),
  trainedModels:   () => http<{ models: TrainedModel[] }>("GET", "/train/models"),
  deleteModel:     (family: string, id: string) =>
    http<{ deleted: string; unregistered: string | null }>("DELETE", `/train/models/${family}/${id}`),
  startTensorboard: () => http<TensorboardStatus>("POST", "/train/tensorboard"),
  tensorboardStatus: () => http<TensorboardStatus>("GET", "/train/tensorboard"),
  arenaUlti:       (req: ArenaRequest) => http<ArenaResponse>("POST", "/arena/ulti", req),
  betliHuNew:      (req: BetliHuNewRequest)      => http<BetliHuState>("POST",   "/betli_hu/new", req),
  betliHuMove:     (req: BetliHuMoveRequest)     => http<BetliHuState>("POST",   "/betli_hu/move", req),
  betliHuState:    (req: BetliHuStateRequest)    => http<BetliHuState>("POST",   "/betli_hu/state", req),
  betliHuAnalysis: (req: { game_id: string })    => http<BetliHuAnalysis>("POST", "/betli_hu/analysis", req),
  betliHuDelete:   (game_id: string)             => http<{ deleted: boolean }>("DELETE", `/betli_hu/session/${game_id}`),
  playNew:    (req: PlayNewRequest)                        => http<PlayState>("POST", "/play/new", req),
  playBid:    (req: PlayBidRequest)                        => http<PlayState>("POST", "/play/bid", req),
  playPickup: (game_id: string)                            => http<PlayState>("POST", "/play/pickup", { game_id }),
  playPass:   (game_id: string, discard_ids: number[] = []) => http<PlayState>("POST", "/play/pass", { game_id, discard_ids }),
  playTrump:  (game_id: string, trump: string)             => http<PlayState>("POST", "/play/trump", { game_id, trump }),
  playKontra: (req: { game_id: string; units: string[] }) => http<PlayState>("POST", "/play/kontra", req),
  playAnalysis: (req: { game_id: string }) => http<PlayAnalysis>("POST", "/play/analysis", req),
  playMove:   (req: { game_id: string; card_id: number }) => http<PlayState>("POST", "/play/move", req),
  playState:  (game_id: string)                            => http<PlayState>("POST", "/play/state", { game_id }),
  playDelete: (game_id: string)                            => http<{ deleted: boolean }>("DELETE", `/play/session/${game_id}`),
  puzzleNew:   ()                                            => http<PuzzleState>("POST", "/puzzle/new"),
  puzzleSolve: (req: { game_id: string; discard_ids: number[] }) => http<PuzzleState>("POST", "/puzzle/solve", req),
  puzzleState: (game_id: string)                            => http<PuzzleState>("GET",  `/puzzle/state/${game_id}`),
  puzzleEnd:   (game_id: string)                            => http<PuzzleState>("POST", `/puzzle/end/${game_id}`),
  renameModel:     (family: string, id: string, new_id: string) =>
    http<{ renamed: boolean; id: string; unregistered?: string | null }>(
      "PATCH", `/train/models/${family}/${id}`, { new_id },
    ),
};

export interface TrainedModel {
  family:        "trickster" | "oldtawer";
  id:            string;
  save_dir:      string;
  meta:          Record<string, unknown>;
  size_bytes:    number;
  mtime:         number;
  registered:    boolean;
  registered_as: string | null;
}
