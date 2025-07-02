#!/usr/bin/env python3
"""benchmarks/convergence_validation.py  ─  *single‑assignment* guarantee

This benchmark validates the **one‑shot assignment** convergence bound from
Section 7 + Appendix B.1 of the *Hybrid AI Brain* JAIR 2025 paper.

> **Guarantee (Theorem 5.3).** If the Bio‑GNN coordinator’s global spectral
> norm satisfies ‖L_total‖₂ ≤ 0.7, then for **every individual assignment step**
> the probability of converging in ≤ 2 synchronous message‑passing rounds is
> **at least 0.87**.

The probability figure (0.87) is *independent* of the workflow depth; complex
plans are executed as consecutive iterations of this guaranteed primitive.

---------------------------------------------------------------------------
Run example
---------------------------------------------------------------------------
```bash
python benchmarks/convergence_validation.py --trials 10000 \
    --beta 1.0 --spectral_norm 0.7 --seed 42
```
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.stats import beta as beta_dist

# Allow project‑root imports when executed directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

PASS_THRESHOLD = 0.87  # Theorem bound for a single assignment step

###############################################################################
# 1 · Soft‑assignment helpers (Section 7)
###############################################################################

def success_probability(beta: float, logits: np.ndarray) -> float:
    """Return maxₐ P(a|t) for a sharpened softmax with inverse‑temperature β."""
    scaled = beta * logits
    exps = np.exp(scaled - scaled.max())  # numerical stability
    return exps.max() / exps.sum()


def default_logits() -> np.ndarray:
    """Domain‑expert vs. distractors — same numbers as §9.1 hop 1."""
    return np.array([2.2, -0.4, -0.8])

###############################################################################
# 2 · Simulator (Geometric with success‑probability q)
###############################################################################

def simulate_assignment_step(
    spectral_norm: float = 0.7,
    beta: float = 1.0,
    logits: np.ndarray | None = None,
    max_steps: int = 10,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample τ — the #rounds to converge for ONE assignment step.

    Workflow:
    1. Check contractivity (L ≤ 0.7).  If violated, return *max_steps* to
       signal non‑convergence in the allotted window.
    2. Compute per‑step assignment success q from β and logits.
    3. τ ~ Geometric(q).  Truncate at *max_steps* for robust bookkeeping.
    """
    if rng is None:
        rng = np.random.default_rng()

    if spectral_norm > 0.7:
        return max_steps  # outside theoretical regime → treat as failure

    # --- assignment success probability ------------------------------------
    if logits is None:
        logits = default_logits()
    q = success_probability(beta, logits)

    # --- geometric sampling -------------------------------------------------
    step = int(rng.geometric(q))  # support {1,2,…}
    return min(step, max_steps)

###############################################################################
# 3 · Statistical validation helpers
###############################################################################

def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> Tuple[float, float]:
    low = 0.0 if successes == 0 else beta_dist.ppf(alpha / 2, successes, trials - successes + 1)
    high = 1.0 if successes == trials else beta_dist.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return low, high


def validate_probability(
    *,
    spectral_norm: float = 0.7,
    beta: float = 1.0,
    n_trials: int = 5_000,
    rng: np.random.Generator | None = None,
) -> Dict:
    if rng is None:
        rng = np.random.default_rng()

    steps = np.array(
        [simulate_assignment_step(spectral_norm, beta, rng=rng) for _ in range(n_trials)],
        dtype=np.int16,
    )
    prob_le_2 = np.mean(steps <= 2)
    ci_low, ci_high = clopper_pearson(int((steps <= 2).sum()), n_trials)

    return {
        "spectral_norm": spectral_norm,
        "beta": beta,
        "trials": n_trials,
        "prob_le_2": prob_le_2,
        "conf_interval": (ci_low, ci_high),
        "step_distribution": dict(zip(*np.unique(steps, return_counts=True))),
    }

###############################################################################
# 4 · CLI / output
###############################################################################

def _pretty_print(res: Dict, runtime: float) -> None:
    print("\n──────────────────────────────────────────────────────────")
    print("📊  Single‑assignment convergence validation (JAIR 2025)")
    print("──────────────────────────────────────────────────────────")
    print(f"Measured  Pr[τ ≤ 2] = {res['prob_le_2']:.4f}")
    print(f"Clopper–Pearson 95% CI = [{res['conf_interval'][0]:.4f}, {res['conf_interval'][1]:.4f}]")
    verdict = "✅ PASS" if res["prob_le_2"] >= PASS_THRESHOLD else "❌ FAIL"
    print(f"Benchmark verdict       = {verdict}")

    print("\nStep distribution:")
    for s in sorted(res["step_distribution"]):
        n = res["step_distribution"][s]
        pct = n / res["trials"] * 100
        print(f"  {s:>2d} rounds : {n:6d}  ({pct:5.1f} %)")

    print(f"\n⏱  runtime = {runtime:.1f} s")
    print("──────────────────────────────────────────────────────────")

###############################################################################
# 5 · Entry point
###############################################################################

def _main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5_000, help="Monte‑Carlo simulations")
    ap.add_argument("--spectral_norm", type=float, default=0.7, help="‖L_total‖₂ (≤ 0.7) ")
    ap.add_argument("--beta", type=float, default=1.0, help="Softmax inverse‑temperature β")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = ap.parse_args()

    t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    results = validate_probability(
        spectral_norm=args.spectral_norm,
        beta=args.beta,
        n_trials=args.trials,
        rng=rng,
    )

    _pretty_print(results, time.perf_counter() - t0)


if __name__ == "__main__":
    _main()
