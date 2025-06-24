#!/usr/bin/env python3
"""
tools/parameter_optimizer.py

Utility for numerically optimizing system parameters to meet
theoretical guarantees in the Hybrid AI Brain paper.
"""

from typing import Dict
from scipy.optimize import root_scalar
import numpy as np

class ParameterOptimizer:
    """
    Provides methods to find optimal values for system parameters
    by solving equations derived from theoretical analysis.
    """
    def __init__(self):
        print("ParameterOptimizer initialized.")

    def optimize_memory_decay_rate(
        self,
        target_staleness: float,
        params: Dict[str, float]
    ) -> float:
        """
        Optimizes the memory decay rate (λ_d) to meet a target staleness.

        Solves:   e^(t_f * λ_d) = 1 + (W_max * λ_d) / (λ_t * c_bar)
        [Section 8.3, Eqn. 1 in paper]

        Args:
            target_staleness: Target max memory staleness (t_f, in seconds).
            params: Dictionary with W_max, lambda_t, c_bar.

        Returns:
            Optimized lambda_d.
        """
        print("\nParameterOptimizer: Optimizing memory decay rate (λ_d)...")
        print(f"  - Target staleness (t_f): {target_staleness}s")
        print(f"  - Input parameters: {params}")

        W_max = params['W_max']
        lambda_t = params['lambda_t']
        c_bar = params['c_bar']

        # Function whose root gives the solution
        def func(lambda_d):
            if lambda_d <= 0:
                return 1e6 # Avoid division by zero
            lhs = np.exp(target_staleness * lambda_d)
            rhs = 1 + (W_max * lambda_d) / (lambda_t * c_bar)
            return lhs - rhs

        # Use root_scalar for robust root finding (bracket the root)
        sol = root_scalar(
            func, bracket=[1e-6, 10.0], method='bisect', xtol=1e-6
        )

        if not sol.converged:
            raise RuntimeError("Could not find an optimal λ_d within the search interval.")

        lambda_d_opt = sol.root
        print(f"  - SOLVED: Optimized λ_d = {lambda_d_opt:.6f}")
        return lambda_d_opt

def main():
    """Demonstrates ParameterOptimizer."""
    print("====== Tools: ParameterOptimizer Demo ======")
    optimizer = ParameterOptimizer()

    # Parameters from Section 8.3 (paper)
    memory_params = {
        "W_max": 50.0,
        "lambda_t": 10.0,
        "c_bar": 0.8,
    }
    target_t_f = 2.97  # seconds

    # Run optimization
    lambda_d = optimizer.optimize_memory_decay_rate(target_t_f, memory_params)

    # Verification
    W_max, lambda_t, c_bar = memory_params.values()
    t_f = (1 / lambda_d) * np.log(1 + (W_max * lambda_d) / (lambda_t * c_bar))

    print("\n--- Verification ---")
    print(f"Calculated staleness t_f = {t_f:.6f}s (should ≈ {target_t_f}s)")
    print("Optimization successful!" if abs(t_f - target_t_f) < 1e-3 else "Check parameter values.")

    print("\n====================================================")
    print("✅ parameter_optimizer.py executed successfully!")

if __name__ == "__main__":
    try:
        import scipy
    except ImportError:
        print("ERROR: This tool requires the 'scipy' library.")
        print("Please install it: pip install scipy")
    else:
        main()
