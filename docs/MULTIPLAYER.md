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
POST /live/table/start  {table_id}             (host; empty chairs stay AI)
```

(All routes above are mounted under `/api`.)

## The game itself (stage 2 — BUILT 2026-08-11)

Exactly as designed, and all of it shipped:

- `Session` has `humans: set[int]` (real seats) and `players: {seat: identity}`.
  The AI game IS `humans={sess.seat}` — one engine, golden-locked;
  `_advance_auction`/`_advance_play` pause on human seats and resolve AI ones.
  Empty chairs stay AI, so 1, 2 or 3 people at a table all work.
- Snapshots are viewer-parameterized (`_snapshot(sess, viewer)`): own hand vs
  backs, turn flags, per-viewer result. Bubbles use per-viewer cursors
  (`bubbles_seen`); `sess.rev` bumps on every mutation for poll adoption.
- The web client reuses the entire game UI; a 1s poll during live games feeds
  the same animation machine that action responses feed in solo play.
- Recording: the identical `_record_session` call, human seats carry user_ids.
- The forehand rotates each round (`table_start`); Következő = the host deals.
- Verified end-to-end by `tests/api/test_mp_game.py` (a full 3-human deal
  through the real routes) and `tests/api/test_live.py` (lobby/table flow).

## Staged rollout

- **Stage 1 (BUILT 2026-08-10):** accounts, lobby (presence + chat), tables with
  join/leave/kick/invite, "Játék élőben" on the splash, logged-in AI games
  record user_id.
- **Stage 2 (BUILT 2026-08-11):** the 1-3-human game loop through the ONE engine
  (see above).
- **Stage 3 (open):** abandonment policy (tab closed mid-deal: reclaim the seat
  after N minutes → AI takes over or the deal is voided), per-table chat during
  play, server-side shared match scorecard, spectating, WS transport if polling
  ever feels laggy; auth hardening before a public launch (token expiry, hashed
  browser tokens, min pw 8, hmac.compare_digest).
