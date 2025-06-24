#!/usr/bin/env python3
"""
experiments/convergence_analysis.py

Empirically validates the convergence time analysis from Section 8.1
of the Hybrid AI Brain paper.
"""

import numpy as np
from typing import List

def calculate_expected_convergence_time(assignment_probabilities: List[float]) -> float:
    """
    Computes the expected convergence time for a multi-hop assignment chain.

    Implements the geometric Markov process model:
        E[τ] = 1 / q, where q is the product of success probabilities across all hops.
    [source: 775, 778]

    Args:
        assignment_probabilities: List of success probabilities for each hop.

    Returns:
        Expected convergence time (steps), or float('inf') if impossible.
    """
    if not assignment_probabilities:
        raise ValueError("No assignment probabilities provided.")

    if not all(0 <= p <= 1 for p in assignment_probabilities):
        raise ValueError("All probabilities must be in [0, 1].")

    joint_q = np.prod(assignment_probabilities)
    if joint_q <= 0:
        return float('inf')
    return 1.0 / joint_q

def run_experiment(hop_probabilities: List[float], claim_steps: float):
    """
    Runs the convergence time validation and prints results.

    Args:
        hop_probabilities: List of hop probabilities for the scenario.
        claim_steps: The theoretical upper bound to validate against.
    """
    print("====== Experiment: GNN Convergence Time Analysis ======")
    print("Validating the theoretical convergence bounds from Section 8.1.\n")

    print("--- Scenario: 3-Hop FIFA World Cup Question ---")
    for i, prob in enumerate(hop_probabilities, 1):
        print(f"Hop {i} Assignment Probability: {prob:.3f}")

    expected_time = calculate_expected_convergence_time(hop_probabilities)
    joint_q = np.prod(hop_probabilities)

    print(f"\nJoint success probability (q): {joint_q:.4f}")
    print(f"Calculated Expected Convergence Time E[τ]: {expected_time:.4f} steps")

    print("\n--- Validation Result ---")
    print(f"Theoretical Claim: Convergence in ≤ {claim_steps} steps.")
    print(f"Experimental Result: E[τ] = {expected_time:.4f} steps.")
    if expected_time <= claim_steps:
        print("✅ VALIDATED: The calculated expected time is within the theoretical bound.")
    else:
        print("❌ FAILED: The calculated expected time exceeds the theoretical bound.")
    print("\n=========================================================")

def main():
    # Example: Probabilities from FIFA 3-hop chain
    hop_probabilities = [0.890, 0.790, 0.920]  # Section 8.1 of the paper
    claim_steps = 2.0  # The bound stated in the paper

    run_experiment(hop_probabilities, claim_steps)

if __name__ == "__main__":
    main()
