// The profile — nickname, my recorded games (paginated), analytics, and one-click
// analysis of any past game (milan 2026-08-11). Identity is layered: the device id
// always works (no forced sign-in); a login/DEV_AUTOLOGIN widens the list to the
// account's games and unlocks renaming. The analysis board is the SAME viewer the
// live post-game uses (useAnalysisViewer) — zero duplicated board code.

import { useCallback, useEffect, useRef, useState } from "react";

import { api, type MeGame, type MeStats } from "./api";
import { getAuth, setAuth, type Auth } from "./auth";
import { deviceId } from "./device";
import { AnalysisBoard } from "./playPanels";
import { useAnalysisViewer } from "./useAnalysisViewer";

const PAGE = 15;

function fmtDate(ts: number): string {
  const d = new Date(ts * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getMonth() + 1)}.${p(d.getDate())}. ${p(d.getHours())}:${p(d.getMinutes())}`;
}

const gpCls = (v: number) => (v > 0 ? "is-pos" : v < 0 ? "is-neg" : "");
const gpTxt = (v: number) => `${v > 0 ? "+" : ""}${Math.round(v * 10) / 10}`;

export function Profile({ onExit }: { onExit: () => void }) {
  const [auth, setAuthState] = useState<Auth | null>(getAuth());
  const [stats, setStats] = useState<MeStats | null>(null);
  const [games, setGames] = useState<MeGame[]>([]);
  const [cursorStack, setCursorStack] = useState<(number | null)[]>([null]);
  const [nextCursor, setNextCursor] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nickDraft, setNickDraft] = useState<string | null>(null);   // null = not editing
  const [openingId, setOpeningId] = useState<string | null>(null);
  const triedDev = useRef(false);

  const av = useAnalysisViewer(useCallback((m: string) => setError(m), []));

  // Local convenience: one silent devlogin attempt (404s in prod → stay anonymous).
  useEffect(() => {
    if (auth || triedDev.current) return;
    triedDev.current = true;
    api.authDevLogin()
      .then((r) => { const a = { token: r.token, username: r.username }; setAuth(a); setAuthState(a); })
      .catch(() => {});
  }, [auth]);

  const loadPage = useCallback(async (cursor: number | null) => {
    try {
      const r = await api.meGames(deviceId(), cursor, PAGE);
      setGames(r.games); setNextCursor(r.next_cursor); setError(null);
    } catch (e) { setError(String(e)); }
  }, []);

  useEffect(() => {
    loadPage(cursorStack[cursorStack.length - 1]);
    api.meStats(deviceId()).then(setStats).catch(() => {});
  }, [auth, cursorStack, loadPage]);

  const saveNick = useCallback(async () => {
    if (nickDraft === null || !auth) return;
    try {
      const r = await api.meNickname(nickDraft.trim());
      const a = { ...auth, username: r.username };
      setAuth(a); setAuthState(a); setNickDraft(null); setError(null);
    } catch (e) {
      const m = String(e).match(/\{"detail":"([^"]+)"\}/);
      setError(m ? m[1] : "Nem sikerült átnevezni.");
    }
  }, [nickDraft, auth]);

  const openGame = useCallback(async (g: MeGame) => {
    setOpeningId(g.id); setError(null);
    await av.openWith(() => api.meAnalysis(g.id, deviceId()));
    setOpeningId(null);
  }, [av]);

  if (av.analysisOpen && av.analysis && av.analysisView) {
    return (
      <AnalysisBoard analysis={av.analysis} analysisView={av.analysisView}
                     effectivePlies={av.effectivePlies} branch={av.branch}
                     branching={av.branching} error={error}
                     scrubPly={av.scrubPly} setScrubPly={av.setScrubPly}
                     onAnalysisCardClick={av.branching ? undefined : av.onAnalysisCardClick}
                     onClearBranch={av.onClearBranch} onCloseAnalysis={av.onCloseAnalysis} />
    );
  }

  const name = auth?.username ?? "vendég";
  const winRate = stats?.games ? Math.round(100 * (stats.wins ?? 0) / stats.games) : null;

  return (
    <div className="app profile">
      <header className="live-head">
        <span className="live-title">Profil</span>
        <button className="btn" onClick={onExit}>Vissza</button>
      </header>
      {error && <div className="error live-error">{error}</div>}

      <main className="profile-main">
        <section className="profile-head">
          <div className="profile-avatar">{name[0].toUpperCase()}</div>
          <div className="profile-who">
            {nickDraft === null ? (
              <div className="profile-name">
                {name}
                {auth && (
                  <button className="profile-edit" title="Átnevezés"
                          onClick={() => setNickDraft(auth.username)}>✎</button>
                )}
              </div>
            ) : (
              <div className="profile-name">
                <input className="auth-input profile-nick-input" value={nickDraft}
                       maxLength={20} autoFocus
                       onChange={(e) => setNickDraft(e.target.value)}
                       onKeyDown={(e) => e.key === "Enter" && saveNick()} />
                <button className="btn" onClick={saveNick}>Mentés</button>
                <button className="btn" onClick={() => setNickDraft(null)}>Mégse</button>
              </div>
            )}
            {!auth && <div className="profile-sub">nincs fiók — az eszköz játszmái</div>}
          </div>
        </section>

        {stats && stats.games > 0 && (
          <section className="profile-stats">
            <div className="profile-stat">
              <b>{winRate}%</b>
              <span>győzelem ({stats.wins}/{stats.games})</span>
            </div>
            {stats.as_soloist && stats.as_soloist.n > 0 && (
              <div className="profile-stat">
                <b>{Math.round(100 * stats.as_soloist.made / stats.as_soloist.n)}%</b>
                <span>győzelem bemondóként ({stats.as_soloist.made}/{stats.as_soloist.n})</span>
              </div>
            )}
            <div className="profile-stat">
              <b className={gpCls(stats.gp_total ?? 0)}>{gpTxt(stats.gp_total ?? 0)}</b>
              <span>össz pont</span>
            </div>
            <div className="profile-stat">
              <b className={gpCls(stats.gp_per_game ?? 0)}>{gpTxt(stats.gp_per_game ?? 0)}</b>
              <span>pont / játszma</span>
            </div>
            {stats.as_soloist && (
              <div className="profile-stat">
                <b>{Math.round(100 * stats.as_soloist.n / stats.games)}%</b>
                <span>bemondóként</span>
              </div>
            )}
          </section>
        )}

        {stats?.contracts && stats.contracts.length > 0 && (
          <section>
            <div className="live-section-title">Játékonként</div>
            <div className="profile-scroll profile-scroll--table">
            <table className="profile-table">
              <thead><tr><th>játék</th><th>n</th><th>pont</th><th>pont/db</th></tr></thead>
              <tbody>
                {stats.contracts.map((c) => (
                  <tr key={c.contract}>
                    <td>{c.contract}</td><td>{c.n}</td>
                    <td className={gpCls(c.gp)}>{gpTxt(c.gp)}</td>
                    <td className={gpCls(c.gp_mean)}>{gpTxt(c.gp_mean)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          </section>
        )}

        <section>
          <div className="live-section-title">Játszmáim</div>
          {games.length === 0 ? (
            <div className="live-empty">Még nincs rögzített játszma.</div>
          ) : (
            <div className="profile-games profile-scroll profile-scroll--games">
              {games.map((g) => (
                <button key={g.id} className="profile-game"
                        disabled={openingId !== null || g.contract === "passz"}
                        title={g.contract === "passz" ? "Mindenki passzolt" : "Elemzés megnyitása"}
                        onClick={() => g.contract !== "passz" && openGame(g)}>
                  <span className="profile-game-date">{fmtDate(g.created_at)}</span>
                  <span className="profile-game-contract">{g.contract}</span>
                  <span className="profile-game-role">
                    {g.contract === "passz" ? "—" : g.i_was_soloist ? "játékos" : "védő"}
                  </span>
                  <span className={`profile-game-gp ${gpCls(g.my_gp)}`}>{gpTxt(g.my_gp)}</span>
                  <span className="profile-game-open">
                    {g.contract === "passz" ? "" : openingId === g.id ? "elemzés…" : "→"}
                  </span>
                </button>
              ))}
            </div>
          )}
          <div className="profile-pager">
            <button className="btn" disabled={cursorStack.length <= 1}
                    onClick={() => setCursorStack((s) => s.slice(0, -1))}>Előző</button>
            <span className="muted">{cursorStack.length}. oldal</span>
            <button className="btn" disabled={nextCursor === null}
                    onClick={() => setCursorStack((s) => [...s, nextCursor])}>Következő</button>
          </div>
        </section>
      </main>
    </div>
  );
}
