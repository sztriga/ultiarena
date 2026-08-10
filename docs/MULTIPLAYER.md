# Multiplayer (Játék élőben) — architecture

milan's spec (2026-08-10): a lobby where people gather and chat while the server
runs; anyone can create an **Asztal** for 3; others join from the lobby or get
invited; the table's creator can kick an unwanted joiner; proper accounts with
usernames; games recorded like the AI games; the SAME game UI, but each player
sees from their own perspective and the other two seats are driven by real people.

## The decisions that are hard to change later

**1. One game engine, parameterized by "which seats are humans" — never a second
engine.** The AI-game session (`apps/api/engine.Session` + flows) already runs the
full auction/kontra/play/score correctly and is golden-locked. Multiplayer is the
same state machine with (a) no AI turns to resolve — `_advance_*` simply stops at
every human turn, and with 3 humans that is every turn — and (b) snapshots
parameterized by a **viewer seat** instead of the single `sess.seat`. We extend the
existing engine; we do not fork it. (Lesson already paid for once: the passed-screen
duplicate.)

**2. Transport: short-polling first, WebSockets later, same data model.** Every
live surface (lobby, table, running game) is served by a poll endpoint returning
state + a sequence cursor for events (chat, game events). Polling at 1–2s is
trivially cheap at friends-scale (a poll is a dict read), works unchanged through
the Cloudflare tunnel, needs no connection lifecycle, and the limits layer already
meters it. When we outgrow it, a WS push replaces the *transport* — the state and
cursor model stay. Corollary: nothing may rely on "drain-once" semantics (the AI
game's speech bubbles do — the MP path must use sequence-numbered events with a
per-viewer cursor instead, because three pollers can't share one drain).

**3. Accounts: username + password now, pluggable later.** `data/users.db`
(sqlite, WAL): `users(id, username UNIQUE COLLATE NOCASE, pw_hash, pw_salt,
created_at)` + `tokens(token, user_id, created_at, last_seen)`. Passwords are
scrypt (stdlib, no deps). The client keeps `{token, username}` in localStorage and
sends `Authorization: Bearer <token>` on every API call; the server resolves it to
a user where it matters and ignores it elsewhere. The existing anonymous
`device_id` stays for resume; a login ATTACHES the device to the user. If we later
want magic links or OAuth, only the register/login endpoints change — the token
model and everything downstream stay.

**4. Identity fills in, never restructures.** `games.db` was user-aware from day
one (`players[].user_id`, null so far). A logged-in player's games — vs AI today,
multiplayer later — record their user_id; device/ip remain as fallback identity.
No schema change.

**5. Abuse control stays IP-keyed.** Tokens and device ids are client-controlled;
the rate/in-flight/session caps keep keying on CF-Connecting-IP. Auth endpoints
get a tighter dedicated per-IP budget (they are the brute-force surface).

## Data model (in-memory, like play sessions)

```
LOBBY     members: {user_id: last_seen}          # presence = polling heartbeat, ~15s expiry
          chat: ring buffer of {seq, user, text, ts}, monotonically increasing seq
TABLE     {id, host_id, name, seats: [user_id|None] * 3, invited: {user_id},
           state: "open" | "playing", game_id: None|str, created_at}
```

Rules: anyone seated in the lobby may join an open seat (no approval step — the
host KICKS instead, milan's "subtle" control); the host may invite (invitee sees
the table highlighted); host leaves → host passes to the next occupant, empty
table dissolves; presence expiry frees seats held by vanished players. One table
per player at a time. When the table fills and the host starts, a game session is
created with `humans = {0,1,2}` mapped to the three user_ids, and the table's
members poll the game endpoint instead.

## Endpoints

```
POST /auth/register {username, password, device_id?}   → {token, username}
POST /auth/login    {username, password, device_id?}   → {token, username}
POST /auth/logout   (token)
GET  /auth/me       (token)                            → {username, ...}

POST /live/poll     {chat_after}  (token)  → {members, chat, tables, my_table, invites}
POST /live/chat     {text}        (token)
POST /live/table/create               → you sit at seat 0, you are host
POST /live/table/join   {table_id}    → first free seat
POST /live/table/leave  {table_id}
POST /live/table/kick   {table_id, user_id}    (host only)
POST /live/table/invite {table_id, username}   (host only)
POST /live/table/start  {table_id}             (host, table full)   [stage 2]
```

## The game itself (stage 2 — the next work item)

- `Session` grows `humans: set[int]` (real seats) and `players: {seat: user_id}`.
  The AI game is `humans={sess.seat}` — the existing behavior, untouched, golden-
  locked. `_advance_auction`/`_advance_play` skip AI resolution for human seats;
  with all three human they are pure rule-keepers (turn order, kontra offers,
  terminal detection, scoring).
- Snapshots take `viewer: int`. Everything already derives from "the human's
  seat"; the change is threading the parameter (own hand vs backs, turn flags,
  result.human_gp per viewer). Bubbles become sequence-numbered events with a
  per-viewer cursor (decision 2).
- The web client reuses the entire game UI. New adoption path: a 1s poll during
  MP games feeds the same animation machine that action responses feed today
  (opponents' moves arrive by poll instead of by my-request-response).
- Kontra popup, marriage bubbles, terített reveal, scoring — all already
  viewer-relative concepts in the engine; they follow the viewer parameter.
- Recording: identical `record_game` call, three human players with user_ids.
- Timeouts/abandonment (someone closes the tab mid-game) get a table-level
  policy later (reclaim the seat after N minutes → AI takes over or the deal is
  voided); v1 documents it as unhandled.

## Staged rollout

- **Stage 1 (BUILT 2026-08-10):** accounts, lobby (presence + chat), tables with
  join/leave/kick/invite, "Játék élőben" on the splash, logged-in AI games
  record user_id. Start button visible when full, disabled ("hamarosan").
- **Stage 2:** the 3-human game loop (engine parameterization + viewer
  snapshots + poll adoption in the client). This is the next work item.
- **Stage 3:** polish — reconnection windows, spectating, per-table chat,
  WS transport if polling ever feels laggy, magic-link auth if passwords annoy.
