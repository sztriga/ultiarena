// UI GOLDEN for the Play tab — the frontend twin of tests/golden.
//
// A real recorded game (actual backend responses, captured over HTTP) is replayed
// through a mocked `api`; this test drives PlayVsAI exactly the way the recording
// was made — open, bid the lowest rung, pass twice, kontra everything, play the
// first legal card ten times — and snapshots the rendered DOM at every phase.
//
// Purpose: make component refactors PROVABLY invisible. The fixture mock throws on
// any deviation from the recorded endpoint sequence (behaviour), and the DOM
// snapshots pin the rendering (structure + classes). If a split changes either,
// this fails — no need to eyeball the UI.
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { makeFixtureApi } from "./fixtureApi";

// vi.mock is hoisted above imports — build the fixture inside the async factory
// and expose it through a global handle for the test body.
vi.mock("../ui/api", async () => {
  const { makeFixtureApi } = await import("./fixtureApi");
  const f = makeFixtureApi();
  (globalThis as Record<string, unknown>).__fixture = f;
  return { api: f.api };
});
const fixture = () =>
  (globalThis as Record<string, unknown>).__fixture as ReturnType<typeof makeFixtureApi>;

import { PlayVsAI } from "../ui/PlayVsAI";
import { cardLabel } from "../ui/cards";
import type { Card } from "../ui/cards";
import { SUITS, RANKS } from "./deck";

function cardById(id: number): Card {
  return { id, suit: SUITS[Math.floor(id / 8)], rank: RANKS[id % 8] } as Card;
}

/** Flush the play-phase animation timers until the board settles. */
async function settle() {
  for (let i = 0; i < 12; i++) {
    await act(async () => {
      vi.advanceTimersByTime(2200);
      await Promise.resolve();
    });
  }
}

describe("PlayVsAI golden (recorded game seat0/seed14)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
    cleanup();
  });

  it("plays the recorded game through every phase with a stable DOM", async () => {
    const { container } = render(<PlayVsAI />);
    // synchronous fireEvent (not user-event): immune to fake-timer deadlocks, and
    // clicks are all this driver needs.
    const click = async (el: Element) => act(async () => { fireEvent.click(el); });

    // ── splash ──
    expect(screen.getByText("Játék a gép ellen")).toBeInTheDocument();
    expect(screen.getByText("Villámtalon")).toBeInTheDocument();
    expect(container.innerHTML).toMatchSnapshot("01-splash");

    // ── new game → bid step (holding 12) ──
    await click(screen.getByText("Játék a gép ellen"));
    await settle();
    expect(container.innerHTML).toMatchSnapshot("02-bid-step");

    // ── bid the recorded rung: pick it in the select, click the 2 recorded discards ──
    const bidStep = fixture().steps[1];
    const bidReq = bidStep.request as { rung_index: number; discard_ids: number[] };
    const newResp = fixture().steps[0].response as {
      auction: { legal_bids: { rung_index: number }[] };
    };
    const optionIdx = newResp.auction.legal_bids.findIndex(
      (b) => b.rung_index === bidReq.rung_index);
    expect(optionIdx).toBeGreaterThanOrEqual(0);
    const select = container.querySelector("select.ulti-bid-select") as HTMLSelectElement;
    expect(select).toBeTruthy();
    await act(async () => { fireEvent.change(select, { target: { value: String(optionIdx) } }); });
    for (const cid of bidReq.discard_ids) {
      const label = cardLabel(cardById(cid));
      const imgs = screen.getAllByAltText(label);
      await click(imgs[imgs.length - 1]);       // the bid hand renders last
    }
    await click(container.querySelector(".ulti-bid-confirm") as Element);
    await settle();
    expect(container.innerHTML).toMatchSnapshot("03-after-bid");

    // ── the two recorded passes (AI overcalled; we let the contract go) ──
    for (let p = 0; p < 2; p++) {
      const passBtn = container.querySelector(".ulti-bid-accept") as Element;
      expect(passBtn).toBeTruthy();
      await click(passBtn);
      await settle();
    }
    expect(container.innerHTML).toMatchSnapshot("04-play-started");

    // ── play out the recording: moves + the kontra decision, in recorded order ──
    let snapped_kontra = false;
    let moves = 0;
    while (!fixture().done() && fixture().peek().action !== "analysis") {
      const step = fixture().peek();
      const before = fixture().consumed();
      if (step.action === "kontra_double") {
        if (!snapped_kontra) {
          expect(container.innerHTML).toMatchSnapshot("05-kontra-offer");
          snapped_kontra = true;
        }
        // the kontra popup is big unit toggles + Kontra/Tovább (milan 2026-08-02):
        // toggle EVERY unit still unselected (a single unit arrives pre-selected;
        // the recording kontra'd them all), then confirm.
        for (const unitBtn of Array.from(
            container.querySelectorAll(".play-kontra-modal .kontra-unit:not(.is-sel)"))) {
          await click(unitBtn);
        }
        const box = container.querySelector(
          ".play-kontra-modal .kontra-go:not([disabled])") as Element;
        expect(box).toBeTruthy();
        await click(box);
      } else if (step.action === "move") {
        const req = step.request as { card_id: number };
        const label = cardLabel(cardById(req.card_id));
        const imgs = screen.getAllByAltText(label);
        await click(imgs[imgs.length - 1]);
        moves += 1;
        if (moves === 3) expect(container.innerHTML).toMatchSnapshot("06-mid-play");
      } else {
        throw new Error(`unexpected recorded action ${step.action}`);
      }
      await settle();
      if (fixture().consumed() === before) {
        throw new Error(
          `stuck on ${step.action} (step #${before}); ` +
          `idle-text: ${container.querySelector(".play-side-idle")?.textContent} | ` +
          `clickable cards: ${container.querySelectorAll(".card-clickable").length}`);
      }
    }
    expect(moves).toBe(10);
    expect(container.innerHTML).toMatchSnapshot("07-result");

    // ── analysis overlay ──
    await click(screen.getByText("Elemzés"));
    await settle();
    expect(container.innerHTML).toMatchSnapshot("08-analysis");

    // every recorded step consumed
    expect(fixture().done()).toBe(true);
  }, 30000);

  it("splash opens Villámtalon", async () => {
    render(<PlayVsAI />);
    await act(async () => { fireEvent.click(screen.getByText("Villámtalon")); });
    await settle();
    expect(document.body.innerHTML).toContain("puzzle");
  });
});
