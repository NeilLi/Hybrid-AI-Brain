#!/usr/bin/env python3
"""
tools/metrics_collector.py

A utility script for collecting, aggregating, and exporting performance metrics
during simulation runs or benchmarks.
"""

from collections import defaultdict
import json
import os
from typing import Dict, List, Any, Optional

class MetricsCollector:
    """
    Collects, summarizes, and exports key performance metrics from a system run.
    Used to validate empirical results against targets and theoretical guarantees.
    """
    def __init__(self):
        self._data: Dict[str, List[float]] = defaultdict(list)
        print("MetricsCollector initialized.")

    def record(self, metric: str, value: float):
        """Record a single data point for a given metric."""
        self._data[metric].append(value)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Computes summary statistics (mean, min, max, count) for all metrics.

        Returns:
            A dict with summary stats for each metric.
        """
        print("\nMetricsCollector: Calculating summary statistics...")
        result = {}
        for name, values in self._data.items():
            if values:
                result[name] = {
                    "mean": float(sum(values)) / len(values),
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "count": len(values)
                }
        return result

    def export(self, file_path: str, summary: Optional[Dict[str, Any]] = None):
        """
        Exports the summary stats to a JSON file.

        Args:
            file_path: Path for the output JSON file.
            summary: (optional) Pre-computed summary, otherwise uses self.summary()
        """
        data = summary if summary is not None else self.summary()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"MetricsCollector: Exported summary to '{file_path}'.")
        except IOError as e:
            print(f"MetricsCollector ERROR: Could not write file '{file_path}': {e}")

    def reset(self):
        """Clears all collected metrics."""
        self._data.clear()
        print("MetricsCollector: Data cleared.")

def main():
    """Demonstrates usage of the MetricsCollector."""
    print("====== Tools: MetricsCollector Demo ======")
    collector = MetricsCollector()

    # 1. Simulate collecting benchmark metrics (100 iterations)
    print("\n--- Simulating benchmark run (100 iterations) ---")
    import random
    for _ in range(100):
        collector.record("latency", random.uniform(0.2, 0.6))
        collector.record("convergence_steps", random.choice([1.0, 2.0, 2.0, 3.0]))
        collector.record("false_block_rate", random.uniform(1e-5, 2e-4))
        collector.record("memory_staleness", random.uniform(2.5, 3.5))

    # 2. Print summary
    summary = collector.summary()
    print("\n--- Final Metric Summary ---")
    print(json.dumps(summary, indent=2))

    # 3. Export results
    output_path = "data/results/benchmark_summary.json"
    collector.export(output_path, summary=summary)

    print("\n====================================================")
    print("✅ metrics_collector.py executed successfully!")

if __name__ == "__main__":
    main()
