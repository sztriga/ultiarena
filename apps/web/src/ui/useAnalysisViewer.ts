// The post-game analysis viewer — state + derivations for AnalysisBoard, extracted
// VERBATIM from PlayVsAI (2026-08-11) so the profile page can open the same board
// on any recorded game. One implementation: the scrubber, the branch composition,
// the hand filtering that preserves server card order — nothing is duplicated.
import { useCallback, useMemo, useState } from "react";

import { api, type PlayAnalysis } from "./api";
import type { Card } from "./cards";
import { useStepScrubber } from "./useStepScrubber";
import type { EffectivePly, AnalysisView } from "./playChrome";

export function useAnalysisViewer(onError: (msg: string) => void) {
  const [analysis, setAnalysis] = useState<PlayAnalysis | null>(null);
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [scrubPly, setScrubPly] = useState(0);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  // Branch: the FULL alternative line (unchanged prefix + god-PV of the chosen fork), held as a
  // complete ply list so branches COMPOSE — you can fork again off any ply of a branch, as deep as
  // you like. `forkPly` is where the latest fork diverged (for the panel + clear).
  const [branch, setBranch] = useState<{ plies: EffectivePly[]; forkPly: number; value: number } | null>(null);
  const [branching, setBranching] = useState(false);

  const openWith = useCallback(async (fetch: () => Promise<PlayAnalysis>) => {
    setAnalysisLoading(true);
    try {
      const ana = await fetch();
      setAnalysis(ana); setScrubPly(ana.per_ply.length); setBranch(null); setAnalysisOpen(true);
    } catch (e) { onError(String(e)); }
    finally { setAnalysisLoading(false); }
  }, [onError]);
  const onCloseAnalysis = useCallback(() => { setAnalysisOpen(false); setBranch(null); }, []);
  const onClearBranch = useCallback(() => {
    if (!branch || !analysis) return;
    setBranch(null); setScrubPly(Math.min(branch.forkPly, analysis.per_ply.length));
  }, [branch, analysis]);

  const effectivePlies = useMemo<EffectivePly[]>(() => {
    if (!analysis) return [];
    if (branch) return branch.plies;                    // the branch already IS the full line
    return analysis.per_ply.map((p, i) => ({
      ply_index: i, player_id: p.player_id, chosen_card: p.chosen_card,
      legal_card_ids: p.legal_card_ids, verdict: p, by_ai: p.by_ai, is_branch: false }));
  }, [analysis, branch]);

  useStepScrubber({ enabled: analysisOpen && analysis !== null, max: effectivePlies.length, setStep: setScrubPly });

  const analysisView = useMemo<AnalysisView | null>(() => {
    if (!analysis) return null;
    // Track what each player has played, then derive hands by FILTERING the initial
    // hands the API sent. Filtering preserves order, so the server's card order (the
    // one rule, ulti.card.sort_hand) survives scrubbing — nothing is re-sorted here.
    const played: Set<number>[] = [new Set(), new Set(), new Set()];
    let trick: { player_id: 0 | 1 | 2; card: Card }[] = [];
    for (let i = 0; i < scrubPly && i < effectivePlies.length; i++) {
      const p = effectivePlies[i];
      if (trick.length === 3) trick = [];
      played[p.player_id].add(p.chosen_card.id);
      trick.push({ player_id: p.player_id, card: p.chosen_card });
    }
    let activePlayer: 0 | 1 | 2 | null = null;
    let legalIds: Set<number> | null = null;
    let branchAtPly: number | null = null;
    if (scrubPly > 0) {
      const last = effectivePlies[scrubPly - 1];
      activePlayer = last.player_id;
      legalIds = new Set<number>(last.legal_card_ids);
      branchAtPly = scrubPly - 1;
      // The just-played card lives on the TABLE only — it used to be restored into
      // the hand too ("click an alternative to fork"), which put it in two places
      // at once and made the middle run one step ahead of the hands (milan spotted
      // the desync 2026-08-02). Forking never needed it: alternatives are still in
      // the hand and the click handler checks legal_card_ids, not hand membership.
    }
    const hands: Card[][] = analysis.initial_hands.map(
      (h, pid) => h.filter((c) => !played[pid].has(c.id)));
    return { hands, currentTrick: trick, activePlayer, legalIds, branchAtPly,
             currentPly: scrubPly, thisPly: scrubPly > 0 ? effectivePlies[scrubPly - 1] : null };
  }, [analysis, scrubPly, effectivePlies]);

  const onAnalysisCardClick = useCallback(async (card: Card) => {
    if (!analysis || !analysisView || branching) return;
    const branchAt = analysisView.branchAtPly;
    if (branchAt === null) return;
    const at = effectivePlies[branchAt];
    if (!at.legal_card_ids.includes(card.id)) return;
    setBranching(true);
    try {
      // moves = the CURRENT line up to the fork (may itself run through earlier branches), so a
      // fork off a branch replays correctly on the backend.
      const moves = effectivePlies.slice(0, branchAt).map((p) => p.chosen_card.id);
      const resp = await api.pisExplore({
        hands: analysis.initial_hands.map((h) => h.map((c) => c.id)),
        soloist: analysis.soloist, starting_leader: analysis.leader, total_tricks: 10,
        moves, forced_card_id: card.id,
        contract: analysis.solve_contract, build_contract: analysis.build_contract,
        trump: analysis.trump,
        talon: analysis.talon.map((c) => c.id),
        declare_marriages: analysis.declare_marriages,
        marriage_restrict: analysis.marriage_restrict,
        multi_weights: analysis.multi_weights,
      });
      // alt_pv[0] IS the forced card; splice the new god-continuation onto the unchanged prefix →
      // the full line. Keeps ply_index == array position, so it composes for the next fork.
      const prefix = effectivePlies.slice(0, branchAt);
      const pvPlies: EffectivePly[] = resp.alt_pv.map((s, j) => ({
        ply_index: branchAt + j, player_id: s.player_id as 0 | 1 | 2, chosen_card: s.card,
        legal_card_ids: s.legal_card_ids, verdict: null, by_ai: false, is_branch: true }));
      setBranch({ plies: [...prefix, ...pvPlies], forkPly: branchAt, value: resp.value });
      setScrubPly(branchAt + 1);
    } catch (e) { onError(String(e)); }
    finally { setBranching(false); }
  }, [analysis, analysisView, effectivePlies, branching, onError]);

  return { analysis, analysisOpen, analysisLoading, branch, branching,
           scrubPly, setScrubPly, effectivePlies, analysisView,
           openWith, onCloseAnalysis, onClearBranch, onAnalysisCardClick };
}
