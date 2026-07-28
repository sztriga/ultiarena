// Keyboard scrubbing: ←/→ to step, Home/End to jump to ends.
// Ignores keypresses while typing in form controls so the seed inputs work.

import { useEffect } from "react";

export function useStepScrubber(opts: {
  enabled: boolean;
  max:     number;
  setStep: (updater: (s: number) => number) => void;
}): void {
  const { enabled, max, setStep } = opts;
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement)  return;
      if (e.target instanceof HTMLSelectElement) return;
      if (e.key === "ArrowRight")     setStep((s) => Math.min(s + 1, max));
      else if (e.key === "ArrowLeft") setStep((s) => Math.max(s - 1, 0));
      else if (e.key === "Home")      setStep(() => 0);
      else if (e.key === "End")       setStep(() => max);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [enabled, max, setStep]);
}
