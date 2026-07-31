// A mock `api` that replays a RECORDED game (real backend responses captured over
// HTTP by the fixture script) in strict order. The UI golden test drives the
// component; every api call must match the recorded endpoint sequence, and gets
// the recorded response. Any deviation — wrong endpoint, extra call — throws,
// so the test fails loudly if a refactor changes the component's behaviour.
import fixture from "./fixtures/game_s0_seed14.json";
import puzzleFixture from "./fixtures/puzzle_new.json";

interface Step {
  action: string;
  endpoint: string;
  request: unknown;
  response: unknown;
}

export function makeFixtureApi() {
  const steps: Step[] = (fixture as { steps: Step[] }).steps;
  let cursor = 0;

  function next(endpoint: string): unknown {
    const step = steps[cursor];
    if (!step) throw new Error(`fixture exhausted; unexpected call to ${endpoint}`);
    if (step.endpoint !== endpoint) {
      throw new Error(
        `call #${cursor}: expected ${step.endpoint} (${step.action}), component called ${endpoint}`);
    }
    cursor += 1;
    return step.response;
  }

  const api = {
    playNew: async () => next("/play/new"),
    playBid: async () => next("/play/bid"),
    playPass: async () => next("/play/pass"),
    playKontra: async () => next("/play/kontra"),
    playMove: async () => next("/play/move"),
    playAnalysis: async () => next("/play/analysis"),
    playPickup: async () => next("/play/pickup"),
    playTrump: async () => next("/play/trump"),
    playState: async () => next("/play/state"),
    playDelete: async () => ({ deleted: true }),
    pisExplore: async () => { throw new Error("pisExplore not in this fixture"); },
    puzzleNew: async () => puzzleFixture,
    puzzleSolve: async () => { throw new Error("puzzleSolve not in this fixture"); },
    puzzleState: async () => puzzleFixture,
    puzzleEnd: async () => puzzleFixture,
    health: async () => ({ status: "ok" }),
  };

  return {
    api,
    steps,
    /** The recorded step the component is ABOUT to trigger. */
    peek: () => steps[cursor],
    done: () => cursor >= steps.length,
    consumed: () => cursor,
  };
}
