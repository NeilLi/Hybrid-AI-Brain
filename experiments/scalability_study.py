#!/usr/bin/env python3
"""
experiments/scalability_study.py

Validates the analytical scalability model from Section 9.2 of the Hybrid AI Brain paper,
and empirically identifies the optimal agent swarm size for parallelized task execution.
"""

import numpy as np

def calculate_processing_time(
    n_agents: np.ndarray,
    T_single: float,
    O_coord: float,
    c_comm: float,
    include_comm: bool = True
) -> np.ndarray:
    """
    Computes total system processing time as a function of swarm size, per the model:
    Time = T_single/n + O_coord + [c_comm * n * log2(n)]  (if include_comm)
    
    Args:
        n_agents: Array of swarm sizes (n values).
        T_single: Time for one agent to complete the entire task solo.
        O_coord: Fixed coordination overhead (e.g., for leader election, graph update).
        c_comm: Communication overhead scaling factor.
        include_comm: If True, adds comm. overhead; otherwise, computes idealized speedup.
    Returns:
        Array of processing times for each swarm size.
    """
    # Prevent division by zero and log(0)
    n_safe = np.maximum(n_agents, 1)
    time_parallel = T_single / n_safe
    comm_overhead = c_comm * n_safe * np.log2(n_safe) if include_comm else 0
    return time_parallel + O_coord + comm_overhead

def main():
    print("====== Experiment: Analytical Scalability Study ======")
    print("Validating the scalability model from Section 9.2 (see Figure 6).\n")

    # Model parameters from the paper [source: 958]
    T_single = 10.0    # Solo agent processing time (seconds)
    O_coord = 0.5      # Fixed coordination overhead (seconds)
    c_comm = 0.1       # Communication overhead factor

    print("--- Model Parameters ---")
    print(f"  - T_single: {T_single}")
    print(f"  - O_coord:  {O_coord}")
    print(f"  - c_comm:   {c_comm}")

    agent_counts = np.arange(1, 21)  # Test n = 1 to 20

    # Compute both models (w/ and w/o communication overhead)
    time_ideal = calculate_processing_time(agent_counts, T_single, O_coord, c_comm, include_comm=False)
    time_realistic = calculate_processing_time(agent_counts, T_single, O_coord, c_comm, include_comm=True)

    # Find optimal n for the realistic model
    optimal_n = int(agent_counts[np.argmin(time_realistic)])
    min_time = float(np.min(time_realistic))

    print("\n--- Scalability Analysis Results ---")
    print(f"  Ideal model time at n=2:      {time_ideal[1]:.4f} s")
    print(f"  Realistic model time at n=2:  {time_realistic[1]:.4f} s")
    print(f"\n  Optimal swarm size (realistic): n_opt = {optimal_n} agents")
    print(f"  Minimum processing time:         {min_time:.4f} seconds")

    # Validate against the theoretical claim
    claim_optimal_n = 6
    is_valid = (optimal_n == claim_optimal_n)
    print("\n--- Validation Result ---")
    print(f"  Theoretical Claim: n_opt = {claim_optimal_n}")
    print(f"  Experimental Result: n_opt = {optimal_n}")
    if is_valid:
        print("  ✅ VALIDATED: The experiment matches the theoretical claim.")
    else:
        print("  ❌ FAILED: The experimental n_opt differs from the theory.")

    # Optionally: output arrays for visualization
    print("\n--- Data for Plotting (n, time_realistic) ---")
    for n, t in zip(agent_counts, time_realistic):
        print(f"    n={n:2d}: {t:.4f} s")
    print("\nYou can plot these results using tools/visualization.py.\n")
    print("=======================================================")

if __name__ == "__main__":
    main()
