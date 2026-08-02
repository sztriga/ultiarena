// Anonymous browser identity — a uuid minted once and kept in localStorage.
//
// This is the "player" until real accounts exist: new games are tagged with it, the
// splash lists/resumes games by it, and the game database records it next to the seat.
// When auth lands, a login simply maps this device id onto a user_id — the plumbing
// (tag on create, list by identity) stays exactly the same.
//
// It is IDENTITY, not security: the server never trusts it for rate/session caps
// (those key on the client IP), and game ids themselves remain unguessable capability
// tokens — knowing someone's device id alone reveals nothing you could fetch.

const KEY = "ultiarena_device";

export function deviceId(): string {
  try {
    let id = localStorage.getItem(KEY);
    if (!id || !/^[0-9a-fA-F-]{8,64}$/.test(id)) {
      id = crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now().toString(16)}-${Math.random().toString(16).slice(2, 10)}`;
      localStorage.setItem(KEY, id);
    }
    return id;
  } catch {
    // Storage unavailable (private mode etc.) → a per-page-load id: resume across
    // reloads won't work, but play is unaffected.
    return `${Date.now().toString(16)}-${Math.random().toString(16).slice(2, 10)}`;
  }
}
