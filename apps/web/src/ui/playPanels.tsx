// Presentational panels extracted from PlayVsAI. Pure props-in, JSX-out — all
// game-flow state stays in the parent. DOM is byte-identical to the inlined
// originals (pinned by the UI golden snapshots).
import type React from "react";

import type { PlayAnalysis, PlayLegalBid, PlayOngoing, PlayResult, PlayState } from "./api";
import type { Card } from "./cards";

type Seat = 0 | 1 | 2;
import { UltiTable, TalonStrip, type SeatChrome } from "./UltiTable";
import { CardChip, KONTRA_WORD, PLAYER_LABEL, TRUMP_LABEL,

         type AnalysisView, type EffectivePly } from "./playChrome";

// GP severity → colour. The verdict shown to the player is the GP the move COST, not the
// solver's internal value: the solver optimises binary indicators (parti ±1, ulti 0/1)
// and 81% of mid-game positions tie in those units, so a mark built on them fires on
// things that cost nothing. `knowable` is the share of that cost that was findable from
// the mover's own seat — a big cost with a small knowable is bad luck, not a blunder.
const SEV_COLOR: Record<string, string> = {
  "baklövés": "#ff6b6b", "hiba": "#ff8a8a", "pontatlanság": "#ffd479", ok: "#9fe3a5",
};
const sevColor = (sv?: string | null) => SEV_COLOR[sv ?? "ok"] ?? "#9fe3a5";
const gp = (x?: number | null) => (x === null || x === undefined ? "—" : `${x > 0 ? "+" : ""}${x.toFixed(1)}`);
// GP column colour, driven by what the move COST its mover (always >= 0). Best move ->
// green; deepening red with the loss. Bands match _SEVERITY on the server so the colour
// and the word never disagree.
// The GP column is an evaluation bar read from the VIEWER's seat, so the number itself
// is neutral (white) and only the SWING is coloured: green when the position improved
// for you, red when it got worse. Deeper shades for bigger swings.
const swingColor = (d: number) => {
  const a = Math.abs(d);
  if (a < 0.05) return "#8b949e";
  if (d > 0) return a >= 8 ? "#4ade80" : a >= 3 ? "#7fd48a" : "#a7dfb0";
  return a >= 8 ? "#ff5252" : a >= 3 ? "#ff8a8a" : "#ffb3b3";
};

export function ResultPanel({ r, withAnalysis, loading, analysisLoading, hasRounds,
                              onPlayAgain, onOpenAnalysis, onShowCard, onAbandon }: {
  r: PlayResult;
  withAnalysis: boolean;
  loading: boolean;
  analysisLoading: boolean;
  hasRounds: boolean;
  onPlayAgain: (rotate?: boolean) => void;
  onOpenAnalysis: () => void;
  onShowCard: () => void;
  onAbandon: () => void;
}) {
  // No sentences here: the signed number IS the message; everything else is a chip.
  // passz rounds rotate the 12-holder like any other round (milan 2026-08-02).
  const sign = r.human_gp > 0 ? "+" : r.human_gp < 0 ? "−" : "";
  return (
    <div className={`play-side-inner play-side-result ${r.user_won ? "is-win" : "is-loss"}`}>
      <div className="res-gp">
        {sign}{Math.abs(r.human_gp)}<span className="res-gp-unit">pont</span>
      </div>
      <div className="res-tags">
        <span className="res-tag">{r.contract}</span>
        {r.kontra_level > 0 && <span className="res-tag res-tag--k">{KONTRA_WORD[r.kontra_level]}</span>}
        {(r.silents ?? []).map((s, i) => (
          <span key={i} className="res-tag res-tag--s">{s.label} {s.gp >= 0 ? "+" : ""}{s.gp}</span>
        ))}
      </div>
      <div className="play-side-actions">
        <button className="betli-hu-deal-btn betli-hu-deal-btn--sm" onClick={() => onPlayAgain(true)} disabled={loading}>
          {loading ? "Osztás…" : "Következő"}
        </button>
        <button className="btn" onClick={onOpenAnalysis} disabled={!withAnalysis || analysisLoading}>
          {analysisLoading ? "Elemzés…" : "Elemzés"}
        </button>
        {hasRounds && (
          <button className="btn" onClick={() => onShowCard()} disabled={loading}>Pontszámok</button>
        )}
        <button className="btn" onClick={onAbandon} disabled={loading}>Új játszma</button>
      </div>
    </div>
  );
}

// Terse on purpose (milan 2026-08-02: no wordy rows) — "most" / "5p" / "2ó".
function agoLabel(idleS: number): string {
  if (idleS < 60) return "most";
  if (idleS < 3600) return `${Math.round(idleS / 60)}p`;
  return `${Math.round(idleS / 3600)}ó`;
}

export function Splash({ loading, error, onNew, onLive, onPuzzle, ongoing = [], onResume, onDiscard }: {
  loading: boolean;
  error: string | null;
  onNew: () => void;
  onLive?: () => void;
  onPuzzle: () => void;
  ongoing?: PlayOngoing[];
  onResume?: (gameId: string) => void;
  onDiscard?: (gameId: string) => void;
}) {
  return (
    <div className="app betli-hu-splash">
      <main className="main betli-hu-splash-main">
        <section style={{ textAlign: "center" }}>
          <h1 className="betli-hu-title">Ulti</h1>
          <div className="splash-actions">
            <button className="betli-hu-deal-btn" onClick={onNew} disabled={loading}>
              {loading ? "…" : "Játék a gép ellen"}
            </button>
            {onLive && (
              <button className="betli-hu-deal-btn betli-hu-deal-btn--live" onClick={onLive}>
                Játék élőben
              </button>
            )}
            <button className="betli-hu-deal-btn betli-hu-deal-btn--ghost"
                    onClick={onPuzzle}>
              Villámtalon
            </button>
          </div>
          {ongoing.length > 0 && onResume && (
            <div className="splash-ongoing">
              <div className="splash-ongoing-title">Folyamatban lévő játékaid</div>
              {ongoing.map(g => (
                <div key={g.game_id} className="splash-ongoing-item">
                  <button className="splash-ongoing-main" disabled={loading}
                          onClick={() => onResume(g.game_id)}>
                    <span className="splash-ongoing-name">{g.contract ?? "licit"}</span>
                    <span className="splash-ongoing-meta">{agoLabel(g.idle_s)}</span>
                  </button>
                  {onDiscard && (
                    <button className="splash-ongoing-x" title="Játék törlése" disabled={loading}
                            onClick={() => onDiscard(g.game_id)}>×</button>
                  )}
                </div>
              ))}
            </div>
          )}
          {error && <div className="error" style={{ marginTop: 14, maxWidth: 480, margin: "14px auto 0" }}>{error}</div>}
        </section>
      </main>
    </div>
  );
}


export function KontraBox({ kontra, contract, trump, loading, kontraSel,
                            onKontra, toggleKontraUnit }: {
  kontra: NonNullable<PlayState["kontra"]>;
  contract: string | null | undefined;
  trump: string | null | undefined;
  loading: boolean;
  kontraSel: Set<string>;
  onKontra: (units: string[]) => void;
  toggleKontraUnit: (key: string) => void;
}) {
  // No prose — a kicker, the contract for context, and one big toggle button per
  // kontra-able unit (a single unit arrives pre-selected, so confirming stays one
  // click). The unit buttons ARE the interface.
  const word = kontra.role === "sol" ? "Rekontra" : "Kontra";
  const units = kontra.units ?? [];
  return (
    <div className="kontra-panel">
          <div className="kontra-kicker">{word}</div>
          <div className="kontra-context">
            {contract}{trump ? ` · ${TRUMP_LABEL(trump)}` : ""}
          </div>
          <div className="kontra-units">
            {units.map((u) => (
              <button key={u.key} type="button" disabled={loading}
                      className={`kontra-unit ${kontraSel.has(u.key) ? "is-sel" : ""}`}
                      onClick={() => toggleKontraUnit(u.key)}>
                {u.label}
              </button>
            ))}
          </div>
          <div className="kontra-actions">
            <button className="kontra-go" onClick={() => onKontra([...kontraSel])}
                    disabled={loading || kontraSel.size === 0}>
              {word}{units.length > 1 && kontraSel.size > 0 ? ` (${kontraSel.size})` : ""}
            </button>
            <button className="btn kontra-skip" onClick={() => onKontra([])} disabled={loading}>
              Tovább
            </button>
          </div>
        </div>
  );
}


// The post-game analysis overlay (god ratings, scrubber, branch exploration) —
// extracted verbatim from PlayVsAI's early-return branch; all state stays there.
export function AnalysisBoard({ analysis, analysisView, effectivePlies, branch,
                                branching, error, scrubPly, setScrubPly,
                                onAnalysisCardClick, onClearBranch, onCloseAnalysis }: {
  analysis: PlayAnalysis;
  analysisView: AnalysisView;
  effectivePlies: EffectivePly[];
  branch: { plies: EffectivePly[]; forkPly: number; value: number } | null;
  branching: boolean;
  error: string | null;
  scrubPly: number;
  setScrubPly: React.Dispatch<React.SetStateAction<number>>;
  onAnalysisCardClick: ((card: Card) => void) | undefined;
  onClearBranch: () => void;
  onCloseAnalysis: () => void;
}) {

    const a = analysisView;
    const v = a.thisPly;
    const maxPly = effectivePlies.length;
    const anaHpi = (analysis.human_play_index ?? 0) as Seat;
    // The GP column is one continuous evaluation read from YOUR seat: take the soloist's
    // position value and flip it once if you are defending, rather than flipping per row
    // by whoever moved. Then the swing between consecutive plies is signed the way a
    // player expects — up is good for you, whoever caused it.
    const evalSeries: Record<number, { val: number | null; delta: number }> = {};
    {
      const vSign = anaHpi === 0 ? 1 : -1;
      let prev: number | null = null;
      for (const p of effectivePlies) {
        const raw = p.verdict?.gp_sol_after;
        const val = raw === null || raw === undefined ? null : raw * vSign;
        evalSeries[p.ply_index] = { val, delta: val === null || prev === null ? 0 : val - prev };
        if (val !== null) prev = val;
      }
    }
    const anaSeat = (pid: Seat): SeatChrome => ({   // same chrome as the game phase (roleTag)
      label: (
        <>
          <span className={`ulti-role-tag ${pid === 0 ? "ulti-role-soloist" : "ulti-role-defender"}`}>
            {pid === 0 ? "Játékos" : "Védő"}
          </span>
          {pid === anaHpi && <span className="ulti-role-tag" style={{ background: "#3a6", color: "#fff" }}>te</span>}
        </>
      ),
    });
    const anaSeats = { 0: anaSeat(0), 1: anaSeat(1), 2: anaSeat(2) };
    const fmt = (n: number) => (Number.isInteger(n) ? String(n) : n.toFixed(1));
    return (
      <div className="app betli-hu-game play-vs-ai">
        <header className="topbar">
          <div className="brand">
            <div className="title">
              Elemzés — {analysis.contract}
              {analysis.trump && <span className="muted"> · {TRUMP_LABEL(analysis.trump)}</span>}
              {branch && <span className="ulti-role-tag" style={{ background: "#a23ed1", color: "#fff", marginLeft: 8 }}>alt ág</span>}
            </div>
            <div className="subtitle">
              {a.currentPly}. / {maxPly}. lépés · <span className="kbd">←</span> <span className="kbd">→</span> a léptetéshez · kattints egy lapra egy ág kipróbálásához
              {v?.verdict?.severity && v.verdict.severity !== "ok" && (
                <> · <b style={{ color: sevColor(v.verdict.severity) }}>{v.verdict.severity.toUpperCase()}</b> <span className="muted">(−{(v.verdict.gp_loss ?? 0).toFixed(1)} GP)</span></>
              )}
              {branching && <> · <i>számolás…</i></>}
            </div>
          </div>
          <div className="controls">
            <button className="btn" onClick={() => setScrubPly((p) => Math.max(0, p - 1))} disabled={a.currentPly === 0}>←</button>
            <button className="btn" onClick={() => setScrubPly((p) => Math.min(maxPly, p + 1))} disabled={a.currentPly >= maxPly}>→</button>
            {branch && <button className="btn" onClick={onClearBranch}>Ág törlése</button>}
            <button className="btn btn-primary" onClick={onCloseAnalysis}>Vissza</button>
          </div>
        </header>
        <main className="main">
          <section>
            {error && <div className="error">{error}</div>}
            <UltiTable
              hands={[a.hands[0], a.hands[1], a.hands[2]]}
              seats={anaSeats}
              currentTrick={a.currentTrick}
              activePlayer={a.activePlayer}
              legalIds={a.legalIds}
              onCardClick={branching ? undefined : onAnalysisCardClick}
              seatNames={{ 0: "Játékos", 1: "Védő", 2: "Védő" }}
              bottomSeat={anaHpi}
              aboveDefenders={
                <>
                  {analysis.trump && (
                    <div className="row" style={{ justifyContent: "center", marginBottom: 4 }}>
                      <span className={`play-badge ${analysis.trump === "hearts" ? "is-piros" : ""}`}>
                        Adu: {TRUMP_LABEL(analysis.trump)}
                      </span>
                    </div>
                  )}
                  <TalonStrip talon={analysis.talon} />
                </>
              }
            />
            {(v?.verdict || branch) && (
              <div className="panel" style={{ marginTop: 10 }}>
                {v?.verdict ? (
                  <>
                    <div className="ana-verdict">
                      <div className="ana-verdict-head">
                        <span className="ana-verdict-ply">{v.ply_index + 1}. lépés</span>
                        <span className="ana-verdict-who">
                          {v.verdict.player_id === 0 ? "Játékos" : `Védő ${v.verdict.player_id}`}
                        </span>
                        {v.verdict.severity && v.verdict.severity !== "ok" ? (
                          <span className="ana-pill" style={{ background: sevColor(v.verdict.severity) }}>
                            {v.verdict.severity}
                          </span>
                        ) : (
                          <span className="ana-pill ana-pill--ok">pontos</span>
                        )}
                      </div>

                      <div className="ana-verdict-row">
                        <div className="ana-cards">
                          <div className="ana-card-slot">
                            <CardChip card={v.verdict.chosen_card} />
                            <span className="ana-card-gp">{gp(v.verdict.gp_chosen)}</span>
                          </div>
                          {(v.verdict.gp_loss ?? 0) > 0.001 && (
                            <>
                              <span className="ana-arrow">→</span>
                              <div className="ana-card-slot">
                                <CardChip card={v.verdict.gp_best_card ?? v.verdict.god_best_card} />
                                <span className="ana-card-gp">{gp(v.verdict.gp_best)}</span>
                              </div>
                            </>
                          )}
                        </div>

                        {(v.verdict.gp_loss ?? 0) > 0.001 ? (
                          <div className="ana-cost">
                            <div className="ana-cost-main" style={{ color: sevColor(v.verdict.severity) }}>
                              −{(v.verdict.gp_loss ?? 0).toFixed(1)}<span className="ana-cost-unit"> GP</span>
                            </div>
                            {v.verdict.gp_loss_knowable !== null && v.verdict.gp_loss_knowable !== undefined && (() => {
                              const loss = v.verdict.gp_loss ?? 0;
                              const known = v.verdict.gp_loss_knowable ?? 0;
                              const pct = loss > 0 ? Math.max(0, Math.min(100, (known / loss) * 100)) : 0;
                              return (
                                <div className="ana-known">
                                  <div className="ana-known-bar"><div className="ana-known-fill" style={{ width: `${pct}%` }} /></div>
                                  <div className="ana-known-text">
                                    <b>{known.toFixed(1)}</b> látható
                                    {loss - known > 0.05 && <> · <span className="muted">{(loss - known).toFixed(1)} pech</span></>}
                                  </div>
                                </div>
                              );
                            })()}
                          </div>
                        ) : (
                          <div className="ana-cost"><div className="ana-cost-main ana-cost-main--ok">nem került semmibe</div></div>
                        )}
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="panel-title">Alternatív ág · isteni folytatás</div>
                    <div className="muted" style={{ fontSize: 12 }}>
                      Elágazás a(z) {(branch?.forkPly ?? 0) + 1}. lépésnél — innentől minden lépés isteni-optimális. Ág végérték: {fmt(branch?.value ?? 0)}.
                    </div>
                  </>
                )}
              </div>
            )}
          </section>
          <section>
            <div className="panel betli-hu-log-panel">
              <div className="panel-title">
                Lépések · GP a te szemszögedből
                <span className="muted" style={{ fontWeight: 400, fontSize: 11 }}>
                  {" "}· a te sorod kiemelve
                </span>
              </div>
              <div className="betli-hu-log-scroll">
                <table className="play-sc-table" style={{ fontSize: 12, width: "100%" }}>
                  <thead>
                    <tr><th>#</th><th>P</th><th>Lépett</th><th>Legjobb</th><th style={{ textAlign: "right" }}>GP</th><th style={{ textAlign: "right" }}>Δ</th></tr>
                  </thead>
                  <tbody>
                    {effectivePlies.map((p) => {
                      const selected = scrubPly - 1 === p.ply_index;
                      const sev = p.verdict?.severity ?? "ok";
                      const blunder = sev !== "ok";
                      // The human's own rows get a slightly lighter background, so you can
                      // find your moves at a glance without an ·AI tag on every AI row.
                      const isHuman = !p.by_ai && p.player_id === anaHpi;
                      const ev = evalSeries[p.ply_index] ?? { val: null, delta: 0 };
                      const myVal = ev.val;
                      const myDelta = ev.delta;
                      return (
                        <tr key={`${p.ply_index}-${p.is_branch ? "b" : "o"}`}
                            onClick={() => setScrubPly(p.ply_index + 1)}
                            style={{ cursor: "pointer",
                              background: selected ? "rgba(99,200,255,0.16)" : p.is_branch ? "rgba(162,62,209,0.10)" : isHuman ? "rgba(255,255,255,0.055)" : undefined,
                              color: blunder ? sevColor(sev) : undefined, fontWeight: blunder ? 600 : undefined }}>
                          <td className="muted">{p.ply_index + 1}</td>
                          <td style={isHuman ? { fontWeight: 700 } : undefined}>
                            {p.player_id === 0 ? "J" : `V${p.player_id}`}{p.is_branch ? "·alt" : ""}
                          </td>
                          <td><CardChip card={p.chosen_card} /></td>
                          <td>{p.verdict ? <CardChip card={p.verdict.gp_best_card ?? p.verdict.god_best_card} /> : "—"}</td>
                          <td style={{ textAlign: "right", fontWeight: 600 }}>{myVal === null ? "" : gp(myVal)}</td>
                          <td style={{ textAlign: "right", color: swingColor(myDelta), fontWeight: 600 }}>
                            {myVal === null || Math.abs(myDelta) < 0.05 ? "" : gp(myDelta)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        </main>
      </div>
    );
  }


// The auction overlay: current bid, history, and the human's bid controls
// (select + trump badge + confirm / passz / Felveszem). Verbatim from PlayVsAI;
// selection state stays in the parent.
export function AuctionPanel({ auction, seat, selPos, setSelPos, setSelTrump, selBid,
                               discards, canConfirm, loading,
                               onConfirm, onAuctionPass, onPickup, waitingName = null }: {
  auction: NonNullable<PlayState["auction"]>;
  seat: number;
  selPos: number | null;
  setSelPos: (p: number | null) => void;
  setSelTrump: (t: string | null) => void;
  selBid: PlayLegalBid | null;
  discards: Set<number>;
  canConfirm: boolean;
  loading: boolean;
  onConfirm: () => void;
  onAuctionPass: () => void;
  onPickup: () => void;
  waitingName?: string | null;     // live game, someone else's turn: "X gondolkodik…"
}) {
  const kind = selBid?.kind ?? "bid";
  const isHolder = !!auction.is_holder;          // you own the standing bid
  const awaitingBid = !!auction.awaiting_bid;    // bid step: you hold the 12
  const reclaim = !!auction.reclaim;             // forehand post-all-pass last look
  const canPickup = !!auction.can_pickup;        // auction step: may Felveszem
  const hist = auction.history.slice(-4);
  const confirmLabel = kind === "pass" ? "Passzolok" : isHolder ? "Emelek" : "Licitálok";
  return (
      <div className="ulti-auction-overlay">
        <div className="ulti-auction-title">Licitálás</div>
        {waitingName && (
          <div className="ulti-auction-waiting">{waitingName} gondolkodik…</div>
        )}
        {auction.current && (
          <div className="ulti-auction-current">
            <span className="ulti-auction-label">Aktuális licit:</span>{" "}
            <span className="ulti-auction-bid-text">{auction.current.contract}</span>
            {/* Adu color is hidden during the auction — announced only when play begins. */}
            <span className="ulti-auction-holder"> — {PLAYER_LABEL[auction.current.pid]}{auction.current.pid === seat ? " (te)" : ""}</span>
          </div>
        )}

        {hist.length > 0 && (
          <div className="ulti-auction-history">
            {hist.map((h, i) => (
              <div key={i} className="ulti-auction-history-entry">
                <span className="ulti-auction-player">{PLAYER_LABEL[h.pid]}{h.pid === seat ? " (te)" : ""}</span>
                {h.kind === "pass"
                  ? <span className="ulti-auction-action pass">passz</span>
                  : <span className="ulti-auction-action bid">{h.contract}{h.trump ? ` · ${TRUMP_LABEL(h.trump)}` : ""}</span>}
              </div>
            ))}
          </div>
        )}

        {awaitingBid ? (
          /* BID STEP — you hold 12: pick from the ladder + discard 2 */
          <>
            <div className="ulti-bid-phase-info">
              {kind === "pass"
                ? <>Dobj el két lapot a talonba. <b>{discards.size}/2</b></>
                : <>Válassz, és dobj el két lapot. <b>{discards.size}/2</b></>}
            </div>
            <div className="ulti-bid-select-row">
              <select className="ulti-bid-select" value={selPos ?? ""}
                      onChange={(e) => {
                        const pos = Number(e.target.value); setSelPos(pos);
                        const b = auction.legal_bids?.[pos];
                        setSelTrump(b && b.trump_options.length ? b.trump_options[0] : null);
                      }}>
                {(auction.legal_bids ?? []).map((b, i) => (
                  <option key={`${b.kind}:${b.rung_index}:${b.bid_index}`} value={i}>
                    {b.label}{b.kind === "bid" ? ` — ${b.value}p` : ""}
                  </option>
                ))}
              </select>
              {/* No trump buttons here — the color is declared AFTER the auction
                  (Ulti keeps it hidden during bidding). Piros = hearts, colorless = none. */}
              {selBid && kind === "bid" && selBid.trump_options.length === 1 && (
                <span className="ulti-badge is-piros">adu: {TRUMP_LABEL(selBid.trump_options[0])}</span>
              )}
              {selBid && kind === "bid" && selBid.colorless && <span className="ulti-badge">színtelen</span>}
              <button className="ulti-bid-confirm" onClick={onConfirm} disabled={loading || !canConfirm}>
                {loading ? "…" : confirmLabel}
              </button>
            </div>
          </>
        ) : (
          /* AUCTION STEP — you hold 10: pick the talon up to bid, or decline */
          <>
            <div className="ulti-bid-phase-info">
              {reclaim
                ? "Vedd fel a talont és játssz, vagy passzolj."
                : isHolder
                ? "Kezdheted a játékot, vagy emelj."
                : canPickup
                ? "Vedd fel a talont, vagy passzolj."
                : "Passzolsz?"}
            </div>
            <div className="ulti-bid-bottom-row">
              <button className="ulti-bid-accept" onClick={onAuctionPass} disabled={loading}>
                {isHolder ? "Elfogadom" : "Passz"}
              </button>
              {canPickup && (
                <button className="ulti-bid-felvesz" onClick={onPickup} disabled={loading}>Felveszem</button>
              )}
            </div>
          </>
        )}
      </div>
  );
}
