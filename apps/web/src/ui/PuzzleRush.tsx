// Villámtalon — a 3-minute talon-picking puzzle rush.
// 12 cards + an AI-chosen contract; bury the 2 cards that give the strongest hand.
// The value net scores your talon vs the optimal one. Combo multiplier, rising difficulty.
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

import { api } from "./api";
import type { PuzzleState, PuzzleResult } from "./api";
import type { Card, Rank, Suit } from "./cards";
import { cardFromId, SUIT_SYMBOL } from "./cards";
import { HandView } from "./CardView";

const RANK_SHORT: Record<Rank, string> = {
  "7": "7", "8": "8", "9": "9", lower: "J", upper: "Q", king: "K", "10": "10", ace: "A",
};
const SUIT_HU: Record<Suit, string> = { hearts: "piros", acorns: "makk", leaves: "zöld", bells: "tök" };

function Chip({ card }: { card: Card }) {
  return (
    <span className="puzzle-chip">
      <span className={`trump-symbol trump-${card.suit}`}>{SUIT_SYMBOL[card.suit]}</span>
      <span className="puzzle-chip-rank">{RANK_SHORT[card.rank]}</span>
    </span>
  );
}

function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  const ss = Math.floor(s % 60);
  return `${m}:${ss.toString().padStart(2, "0")}`;
}

function multiplier(combo: number): number {
  return 1 + 0.25 * Math.min(combo, 8);
}

export function PuzzleRush({ onExit }: { onExit: () => void }) {
  const [phase, setPhase] = useState<"intro" | "play" | "done">("intro");
  const [state, setState] = useState<PuzzleState | null>(null);
  const [picked, setPicked] = useState<number[]>([]);
  const [flash, setFlash] = useState<PuzzleResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [remain, setRemain] = useState(0);

  const deadlineRef = useRef<number>(0);
  const flashTimer = useRef<number | null>(null);
  const endedRef = useRef(false);
  const busyRef = useRef(false);
  const gidRef = useRef<string | null>(null);
  const endRef = useRef<() => void>(() => {});
  gidRef.current = state?.game_id ?? null;

  const endGame = useCallback(async () => {
    if (endedRef.current) return;
    endedRef.current = true;
    setPhase("done");
    const gid = gidRef.current;
    if (gid) {
      try { setState(await api.puzzleEnd(gid)); } catch { /* keep last snapshot */ }
    }
  }, []);
  endRef.current = endGame;

  const start = useCallback(async () => {
    setLoading(true);
    try {
      const s = await api.puzzleNew();
      deadlineRef.current = performance.now() + s.time_left * 1000;
      endedRef.current = false;
      busyRef.current = false;
      setState(s); setPicked([]); setFlash(null); setRemain(s.time_left);
      setPhase("play");
    } catch (e) {
      setPhase("intro");
    } finally { setLoading(false); }
  }, []);

  const submit = useCallback(async (ids: number[]) => {
    if (busyRef.current || !gidRef.current) return;
    busyRef.current = true;
    try {
      const s = await api.puzzleSolve({ game_id: gidRef.current, discard_ids: ids });
      if (s.last_result) {
        setFlash(s.last_result);
        if (flashTimer.current) window.clearTimeout(flashTimer.current);
        flashTimer.current = window.setTimeout(() => setFlash(null), 1500);
      }
      setState(s); setPicked([]);
      if (s.done) endRef.current();
    } catch { /* ignore transient */ }
    finally { busyRef.current = false; }
  }, []);

  // auto-submit once two cards are picked
  useEffect(() => {
    if (phase === "play" && picked.length === 2 && !busyRef.current) submit(picked);
  }, [picked, phase, submit]);

  // countdown (stable interval per phase; endRef avoids stale closures)
  useEffect(() => {
    if (phase !== "play") return;
    const tick = () => {
      const left = Math.max(0, (deadlineRef.current - performance.now()) / 1000);
      setRemain(left);
      if (left <= 0) endRef.current();
    };
    tick();
    const iv = window.setInterval(tick, 250);
    return () => window.clearInterval(iv);
  }, [phase]);

  // end the session if the player navigates away mid-rush
  useEffect(() => () => {
    if (flashTimer.current) window.clearTimeout(flashTimer.current);
    if (gidRef.current && !endedRef.current) api.puzzleEnd(gidRef.current).catch(() => {});
  }, []);

  const toggle = useCallback((card: Card) => {
    if (busyRef.current) return;
    setPicked((prev) => {
      if (prev.includes(card.id)) return prev.filter((x) => x !== card.id);
      if (prev.length >= 2) return prev;
      return [...prev, card.id];
    });
  }, []);

  const shell = (body: ReactNode) => (
    <div className="app betli-hu-game play-vs-ai puzzle-rush">
      <main className="main puzzle-main">{body}</main>
    </div>
  );

  // ── intro ──
  if (phase === "intro") {
    return shell(
      <section className="panel puzzle-card puzzle-intro">
        <h1 className="betli-hu-title">Villámtalon</h1>
        <p className="puzzle-lead">3 perc. Ahány talont csak meg tudsz oldani.</p>
        <ul className="puzzle-rules">
          <li>Kapsz <b>12 lapot</b> és egy játékot (pl. <b>ulti</b>).</li>
          <li>Tegy le <b>2 lapot</b> — a lehető legerősebb kézért.</li>
          <li>A háló pontozza a döntésed; sorozatért <b>szorzó</b> jár.</li>
          <li>Ahogy nő a pontszám, <b>nehezednek</b> a feladványok.</li>
        </ul>
        <div className="puzzle-actions">
          <button className="betli-hu-deal-btn" onClick={start} disabled={loading}
                  style={{ background: "#7c3aed" }}>
            {loading ? "…" : "Start"}
          </button>
          <button className="btn puzzle-back" onClick={onExit}>Vissza</button>
        </div>
      </section>,
    );
  }

  // ── done ──
  if (phase === "done") {
    return shell(
      <section className="panel puzzle-card puzzle-done">
        <h1 className="betli-hu-title">Idő lejárt!</h1>
        <div className="puzzle-final-score">{state?.score ?? 0}</div>
        <div className="puzzle-final-label">pont</div>
        <div className="puzzle-final-stats">
          <span><b>{state?.solved ?? 0}</b> feladvány</span>
          <span>leghosszabb sorozat <b>{state?.best_combo ?? 0}</b></span>
        </div>
        <div className="puzzle-actions">
          <button className="betli-hu-deal-btn" onClick={start} disabled={loading}
                  style={{ background: "#7c3aed" }}>
            {loading ? "…" : "Újra"}
          </button>
          <button className="btn puzzle-back" onClick={onExit}>Vissza</button>
        </div>
      </section>,
    );
  }

  // ── play ──
  const pz = state?.puzzle ?? null;
  const low = remain <= 30;
  const combo = state?.combo ?? 0;
  return shell(
    <>
      <header className="puzzle-hud">
        <div className={`puzzle-timer ${low ? "is-low" : ""}`}>{fmtTime(remain)}</div>
        <div className="puzzle-score">
          <b>{state?.score ?? 0}</b><span>pont</span>
        </div>
        {combo >= 2 && (
          <div className="puzzle-combo">🔥 ×{multiplier(combo).toFixed(2)}</div>
        )}
        <div className="puzzle-spacer" />
        <div className="puzzle-solved">{state?.solved ?? 0} kész</div>
        <button className="btn puzzle-quit" onClick={endGame}>Feladom</button>
      </header>

      <section className="panel puzzle-stage">
        {pz && (
          <>
            <div className="puzzle-prompt">
              <span className="puzzle-prompt-lead">Legerősebb</span>{" "}
              <span className="puzzle-contract">{pz.contract_label}</span>
              {pz.trump && (
                <span className="puzzle-adu"> — adu:{" "}
                  <b className={`trump-${pz.trump}`}>{SUIT_HU[pz.trump]}</b>
                </span>
              )}
            </div>
            <div className="puzzle-instruction">Tegy le 2 lapot a talonba</div>
            <div className="puzzle-hand">
              <HandView cards={pz.cards} fan selectedIds={new Set(picked)} onCardClick={toggle} />
            </div>
          </>
        )}

        {flash && (
          <div className={`puzzle-flash ${flash.perfect ? "is-perfect" : flash.quality >= 0.85 ? "is-good" : "is-miss"}`}>
            <div className="puzzle-flash-points">+{flash.points}</div>
            <div className="puzzle-flash-msg">
              {flash.perfect ? "Tökéletes!" : flash.quality >= 0.85 ? "Erős talon!" : "Jobb volt:"}
              {!flash.perfect && flash.quality < 0.85 && (
                <span className="puzzle-flash-opt">
                  {flash.optimal_ids.map((id) => <Chip key={id} card={cardFromId(id)} />)}
                </span>
              )}
            </div>
          </div>
        )}
      </section>
    </>,
  );
}
