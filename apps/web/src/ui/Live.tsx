// Játék élőben — login/register, then the lobby: presence, chat, and the Asztal
// (3-seat tables with free join, host kick, invites). Stage 1 of multiplayer
// (docs/MULTIPLAYER.md); the game start itself is stage 2 and the button says so.
//
// Transport is a 2s poll (design decision 2): one request returns presence, the
// chat tail after our cursor, and every table — the whole screen re-renders from
// that single state object.

import { useCallback, useEffect, useRef, useState } from "react";

import { api, type LiveChatMsg, type LivePoll, type LiveTable } from "./api";
import { getAuth, setAuth, type Auth } from "./auth";
import { deviceId } from "./device";

const POLL_MS = 2000;

export function Live({ onExit }: { onExit: () => void }) {
  const [auth, setAuthState] = useState<Auth | null>(getAuth());
  if (!auth) {
    return <AuthPanel onExit={onExit}
                      onAuthed={(a) => { setAuth(a); setAuthState(a); }} />;
  }
  return <Lobby auth={auth} onExit={onExit}
                onLogout={() => { api.authLogout().catch(() => {}); setAuth(null); setAuthState(null); }} />;
}

// ── Login / register ────────────────────────────────────────────────────────────

function AuthPanel({ onAuthed, onExit }: { onAuthed: (a: Auth) => void; onExit: () => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = useCallback(async () => {
    if (busy || username.length < 3 || password.length < 6) return;
    setBusy(true); setError(null);
    try {
      const call = mode === "login" ? api.authLogin : api.authRegister;
      const r = await call({ username, password, device_id: deviceId() });
      onAuthed({ token: r.token, username: r.username });
    } catch (e) {
      // surface only the backend's Hungarian detail, not the raw HTTP noise
      const m = String(e).match(/\{"detail":"([^"]+)"\}/);
      setError(m ? m[1] : "Nem sikerült — próbáld újra.");
    } finally { setBusy(false); }
  }, [mode, username, password, busy, onAuthed]);

  return (
    <div className="app betli-hu-splash">
      <main className="main betli-hu-splash-main">
        <section style={{ textAlign: "center" }}>
          <h1 className="betli-hu-title">Ulti</h1>
          <div className="auth-panel">
            <div className="auth-tabs">
              <button className={`auth-tab ${mode === "login" ? "is-active" : ""}`}
                      onClick={() => setMode("login")}>Belépés</button>
              <button className={`auth-tab ${mode === "register" ? "is-active" : ""}`}
                      onClick={() => setMode("register")}>Regisztráció</button>
            </div>
            <input className="auth-input" placeholder="név" value={username}
                   maxLength={20} autoFocus
                   onChange={(e) => setUsername(e.target.value)}
                   onKeyDown={(e) => e.key === "Enter" && submit()} />
            <input className="auth-input" placeholder="jelszó" type="password" value={password}
                   maxLength={128}
                   onChange={(e) => setPassword(e.target.value)}
                   onKeyDown={(e) => e.key === "Enter" && submit()} />
            <button className="betli-hu-deal-btn auth-submit" onClick={submit}
                    disabled={busy || username.length < 3 || password.length < 6}>
              {busy ? "…" : mode === "login" ? "Belépés" : "Regisztráció"}
            </button>
            {error && <div className="auth-error">{error}</div>}
            <button className="btn auth-back" onClick={onExit}>Vissza</button>
          </div>
        </section>
      </main>
    </div>
  );
}

// ── Lobby ───────────────────────────────────────────────────────────────────────

function Lobby({ auth, onExit, onLogout }: {
  auth: Auth; onExit: () => void; onLogout: () => void;
}) {
  const [state, setState] = useState<LivePoll | null>(null);
  const [chat, setChat] = useState<LiveChatMsg[]>([]);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const cursor = useRef(0);
  const chatEnd = useRef<HTMLDivElement | null>(null);

  const poll = useCallback(async () => {
    try {
      const r = await api.livePoll(cursor.current);
      setState(r);
      if (r.chat.length) {
        cursor.current = r.chat[r.chat.length - 1].seq;
        setChat((c) => [...c, ...r.chat].slice(-200));
      }
      setError(null);
    } catch (e) {
      if (String(e).includes("401")) onLogout();     // token expired/revoked
      else setError("Kapcsolati hiba — újrapróbálkozás…");
    }
  }, [onLogout]);

  useEffect(() => {
    poll();
    const t = window.setInterval(poll, POLL_MS);
    return () => window.clearInterval(t);
  }, [poll]);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ block: "end" });
  }, [chat.length]);

  const act = useCallback(async (fn: () => Promise<unknown>) => {
    try { await fn(); await poll(); }
    catch (e) {
      const m = String(e).match(/\{"detail":"([^"]+)"\}/);
      setError(m ? m[1] : "Nem sikerült.");
    }
  }, [poll]);

  const send = useCallback(() => {
    const text = draft.trim();
    if (!text) return;
    setDraft("");
    act(() => api.liveChat(text));
  }, [draft, act]);

  const myTable = state?.tables.find((t) => t.table_id === state.my_table) ?? null;

  return (
    <div className="app live-lobby">
      <header className="live-head">
        <span className="live-title">Játék élőben</span>
        <span className="live-me">{auth.username}</span>
        <button className="btn" onClick={onLogout}>Kilépés</button>
        <button className="btn" onClick={onExit}>Főoldal</button>
      </header>
      {error && <div className="error live-error">{error}</div>}

      <main className="live-main">
        <section className="live-tables">
          <div className="live-section-title">
            Asztalok
            {!myTable && (
              <button className="betli-hu-deal-btn betli-hu-deal-btn--sm"
                      onClick={() => act(api.tableCreate)}>Új asztal</button>
            )}
          </div>
          {state === null ? (
            <div className="live-empty">Kapcsolódás…</div>
          ) : state.tables.length === 0 ? (
            <div className="live-empty">Még nincs asztal — nyiss egyet!</div>
          ) : (
            state.tables.map((t) => (
              <TableCard key={t.table_id} t={t} mine={t.table_id === state.my_table}
                         seated={myTable !== null} act={act} />
            ))
          )}
        </section>

        <section className="live-side">
          <div className="live-section-title">Lobby ({state?.members.length ?? 0})</div>
          <div className="live-members">
            {(state?.members ?? []).map((m) => (
              <span key={m} className={`live-member ${m === auth.username ? "is-me" : ""}`}>
                {m}
                {myTable && myTable.is_host && !myTable.full && m !== auth.username &&
                  !myTable.seats.some((s) => s?.username === m) && (
                  <button className="live-invite" title="Meghívás az asztalodhoz"
                          onClick={() => act(() => api.tableInvite(myTable.table_id, m))}>+</button>
                )}
              </span>
            ))}
          </div>

          <div className="live-section-title">Csevegés</div>
          <div className="live-chat">
            {chat.map((c) => (
              <div key={c.seq} className="live-chat-msg">
                <span className={`live-chat-user ${c.user === auth.username ? "is-me" : ""}`}>{c.user}</span>
                <span className="live-chat-text">{c.text}</span>
              </div>
            ))}
            <div ref={chatEnd} />
          </div>
          <div className="live-chat-row">
            <input className="auth-input live-chat-input" placeholder="üzenet…" value={draft}
                   maxLength={300}
                   onChange={(e) => setDraft(e.target.value)}
                   onKeyDown={(e) => e.key === "Enter" && send()} />
            <button className="btn" onClick={send} disabled={!draft.trim()}>Küld</button>
          </div>
        </section>
      </main>
    </div>
  );
}

function TableCard({ t, mine, seated, act }: {
  t: LiveTable; mine: boolean; seated: boolean;
  act: (fn: () => Promise<unknown>) => void;
}) {
  return (
    <div className={`live-table ${mine ? "is-mine" : ""} ${t.invited_me ? "is-invited" : ""}`}>
      <div className="live-table-head">
        <span className="live-table-host">{t.host} asztala</span>
        {t.invited_me && !mine && <span className="live-invited-tag">meghívtak!</span>}
      </div>
      <div className="live-seats">
        {t.seats.map((s, i) => (
          <div key={i} className={`live-seat ${s ? "is-taken" : ""} ${s?.is_me ? "is-me" : ""}`}>
            {s ? (
              <>
                <span className="live-seat-name">{s.username}{s.is_host ? " ♔" : ""}</span>
                {mine && t.is_host && !s.is_me && (
                  <button className="live-kick" title="Kitevés az asztaltól"
                          onClick={() => act(() => api.tableKick(t.table_id, s.user_id))}>×</button>
                )}
              </>
            ) : (
              <span className="live-seat-free">szabad</span>
            )}
          </div>
        ))}
      </div>
      <div className="live-table-actions">
        {mine ? (
          <>
            {t.is_host && (
              <button className="betli-hu-deal-btn betli-hu-deal-btn--sm" disabled
                      title="A játszma indítás a következő lépés — hamarosan">
                Indítás {t.full ? "(hamarosan)" : ""}
              </button>
            )}
            <button className="btn" onClick={() => act(() => api.tableLeave(t.table_id))}>Felállok</button>
          </>
        ) : (
          !t.full && !seated && (
            <button className="btn" onClick={() => act(() => api.tableJoin(t.table_id))}>Leülök</button>
          )
        )}
      </div>
    </div>
  );
}
