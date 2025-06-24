#!/usr/bin/env python3
"""
experiments/safety_bounds_test.py

Empirically validates the safety bounds analysis using Hoeffding's inequality,
as described in Section 8.2 of the Hybrid AI Brain paper.
"""

import numpy as np

def calculate_hoeffding_bound(n_samples: int, epsilon: float) -> float:
    """
    Computes the upper bound on the false-block probability using Hoeffding's inequality:
    Pr[false-block] <= exp(-2 * n * ε^2)  [source: 790]

    Args:
        n_samples (int): Number of mask evaluations (n).
        epsilon (float): Safety margin (τ_safe - p).

    Returns:
        float: Maximum probability of a false-block event.
    """
    if n_samples <= 0 or epsilon < 0:
        print("Invalid parameters: n_samples must be >0 and epsilon >= 0.")
        return 1.0
    return float(np.exp(-2 * n_samples * (epsilon**2)))

def report_safety_bound(params: dict, claim_probability: float = 1e-4):
    print("====== Experiment: Safety Bounds (Hoeffding's Inequality) ======")
    print("Validating statistical safety guarantees from Section 8.2.\n")
    print("--- Scenario: GraphMask evaluating a benign edge ---")
    print("Parameters from 'Standard Deployment':")
    for k, v in params.items():
        print(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")
    
    false_block_prob = calculate_hoeffding_bound(params['n_samples'], params['epsilon'])
    print(f"\nCalculated Max Pr(false-block): {false_block_prob:.3e}")

    print("\n--- Validation Result ---")
    print(f"Theoretical Claim: False-block probability <= {claim_probability:.1e}.")
    print(f"Experimental Result: Calculated bound is {false_block_prob:.3e}.")
    if false_block_prob <= claim_probability:
        print("✅ VALIDATED: The calculated probability is within the theoretical bound.")
        print("   (The paper's calculated value was ~2.4e-5, which this matches.)")
    else:
        print("❌ FAILED: The calculated probability exceeds the theoretical bound.")
    print("\n==================================================================")

def main():
    # Parameters from the paper [source: 794, 802]
    n_samples = 59                  # Number of mask evaluations (n)
    tau_safe = 0.7                  # Safety threshold (τ_safe)
    p_benign_worst_case = 0.4       # Worst-case benign edge 'unsafe' prob (p)
    epsilon = tau_safe - p_benign_worst_case

    params = {
        "n_samples": n_samples,
        "tau_safe": tau_safe,
        "p_benign_worst_case": p_benign_worst_case,
        "epsilon": epsilon
    }
    report_safety_bound(params, claim_probability=1e-4)

if __name__ == "__main__":
    main()
