#!/usr/bin/env python3
"""
tools/visualization.py

Utility for plotting and visualizing theoretical models and benchmarks
from the Hybrid AI Brain paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

class Visualizer:
    """
    Provides methods for generating key plots for the Hybrid AI Brain system.
    """

    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        try:
            plt.style.use(style)
        except Exception:
            print(f"Warning: Matplotlib style '{style}' not found. Using default.")
        print("Visualizer initialized.")

    def plot_scalability_model(
        self,
        params: Dict[str, float],
        save_path: str = "data/results/scalability_model.png"
    ):
        """
        Plots the Analytical Scalability Model (Fig. 6, paper).
        Compares simplified (no comm. overhead) and realistic models.

        Args:
            params: Dict with T_single, O_coord, O_comm_factor.
            save_path: Path to save the generated figure.
        """
        T_single = params['T_single']
        O_coord = params['O_coord']
        c = params['O_comm_factor']
        n_agents = np.arange(1, 21)

        # Processing time calculations
        time_simple = T_single / n_agents + O_coord
        time_real = T_single / n_agents + O_coord + c * n_agents * np.log2(n_agents + 1e-9)

        # Find optimal n for realistic model
        optimal_idx = np.argmin(time_real)
        optimal_n = n_agents[optimal_idx]
        min_time = time_real[optimal_idx]

        # --- Plotting ---
        plt.figure(figsize=(10, 6))

        plt.axhline(
            y=T_single, color='red', linestyle='--',
            label=f'Single Agent Baseline (T_single={T_single:.1f}s)'
        )
        plt.plot(n_agents, time_simple, 'o-', label='Simplified Model', color='dodgerblue')
        plt.plot(n_agents, time_real, 'o-', label='Realistic Model', color='navy')

        # Highlight optimal point
        plt.plot(optimal_n, min_time, 'ro', markersize=10, label=f'Optimal Swarm Size (n={optimal_n})')

        plt.title('Analytical Scalability Model\nImpact of Communication Overhead', fontsize=15)
        plt.xlabel('Number of Agents (n)', fontsize=12)
        plt.ylabel('Total Processing Time (seconds)', fontsize=12)
        plt.xticks(n_agents)
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

        try:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            print(f"Visualizer: Scalability plot saved to '{save_path}'.")
        except Exception as e:
            print(f"Visualizer ERROR: Could not save plot: {e}")

        plt.close()

def main():
    """Demo of Visualizer: generates the scalability plot."""
    print("====== Tools: Visualizer Demo ======")
    visualizer = Visualizer()

    # Parameters from Section 9.2
    params = {
        "T_single": 10.0,
        "O_coord": 0.5,
        "O_comm_factor": 0.1,
    }

    import os
    os.makedirs("data/results", exist_ok=True)
    visualizer.plot_scalability_model(params)

    print("\n====================================================")
    print("✅ visualization.py executed successfully!")

if __name__ == "__main__":
    try:
        import matplotlib
    except ImportError:
        print("ERROR: This tool requires 'matplotlib'. Install via: pip install matplotlib")
    else:
        main()
