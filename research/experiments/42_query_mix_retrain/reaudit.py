import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
from ulti.eval.head_audit import audit

ONLY = ["duri_colored", "reach100_20"]


def main():
    print("── DEPLOYED (with isotonic, as served) ──", flush=True)
    old = audit(n_uniform=300, n_argmax=200, only=ONLY, calibrate=True)
    print("── CANDIDATE (mixture retrain + query isotonic) ──", flush=True)
    new = audit(n_uniform=300, n_argmax=200, only=ONLY, calibrate=True,
                weights_dir=str(Path(__file__).parent / "candidate_full"))
    print(f"\n{'head':14s} {'old pred':>9s} {'old god':>8s} | {'new pred':>9s} {'new god':>8s}  (argmax)")
    for h in ONLY:
        print(f"{h:14s} {old[h]['argmax_mean_pred']:9.3f} {old[h]['argmax_mean_god']:8.3f} | "
              f"{new[h]['argmax_mean_pred']:9.3f} {new[h]['argmax_mean_god']:8.3f}")


if __name__ == "__main__":
    main()
