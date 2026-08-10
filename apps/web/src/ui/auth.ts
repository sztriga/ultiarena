// The logged-in account, kept in localStorage next to the device id. The token is
// an opaque bearer credential (apps/api/users.py); api.ts attaches it to every
// request, and the backend simply ignores it where identity doesn't matter.

const KEY = "ultiarena_auth";

export interface Auth {
  token: string;
  username: string;
}

export function getAuth(): Auth | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const a = JSON.parse(raw) as Auth;
    return a.token && a.username ? a : null;
  } catch {
    return null;
  }
}

export function setAuth(a: Auth | null): void {
  try {
    if (a) localStorage.setItem(KEY, JSON.stringify(a));
    else localStorage.removeItem(KEY);
  } catch {
    /* storage unavailable → session-only login */
  }
}
