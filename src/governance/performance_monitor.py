#!/usr/bin/env python3
"""
src/governance/performance_monitor.py

Monitors key system metrics to detect performance drift against manifest thresholds.
"""

import logging
from collections import deque
from typing import Dict, Any, Deque, Optional

logger = logging.getLogger("hybrid_ai_brain.performance_monitor")
logging.basicConfig(level=logging.INFO)

class TelemetrySystem:
    def get_latest_metrics(self) -> Dict[str, float]:
        """Simulates fetching real-time system metrics."""
        import random
        return {
            "latency": random.uniform(0.2, 0.6),              # Expected <= 0.5s
            "convergence_steps": random.choice([1.0, 2.0, 2.0, 3.0]), # <= 2
            "false_block_rate": random.uniform(1e-5, 2e-4),    # <= 1e-4
            "memory_staleness": random.uniform(2.5, 3.5),      # < 3s
        }

class PerformanceMonitor:
    """
    Tracks KPIs and detects drift for governance-layer adaptation.
    """
    def __init__(self, telemetry_system: TelemetrySystem, window_size: int = 10):
        self.telemetry = telemetry_system
        self.window_size = window_size
        self.metric_history: Dict[str, Deque[float]] = {}
        logger.info("PerformanceMonitor initialized.")

    def record_metrics(self) -> None:
        """Pulls and appends new metrics to the history."""
        latest_metrics = self.telemetry.get_latest_metrics()
        logger.info(f"Recording new metrics: { {k: f'{v:.4f}' for k, v in latest_metrics.items()} }")
        for key, value in latest_metrics.items():
            if key not in self.metric_history:
                self.metric_history[key] = deque(maxlen=self.window_size)
            self.metric_history[key].append(value)

    def check_drift(self, performance_targets: Dict[str, Any]) -> bool:
        """
        Checks whether recent metrics (windowed avg) violate the thresholds.
        Returns True if drift is detected.
        """
        if not performance_targets:
            logger.warning("No performance targets defined. Skipping drift check.")
            return False

        logger.info("Checking for performance drift...")
        drift_detected = False

        for metric, history in self.metric_history.items():
            if not history:
                logger.debug(f"No history for metric '{metric}'.")
                continue
            # Accept both 'metric_target' and 'metric' as key (more robust)
            target_key = f"{metric}_target" if f"{metric}_target" in performance_targets else metric
            target_value = performance_targets.get(target_key)
            if target_value is None:
                logger.warning(f"No target defined for metric '{metric}'.")
                continue

            current_avg = sum(history) / len(history)
            if current_avg > target_value:
                logger.warning(
                    f"DRIFT DETECTED on '{metric}': Avg {current_avg:.4f} > Target {target_value:.4f}"
                )
                drift_detected = True
            else:
                logger.info(
                    f"'{metric}': Avg {current_avg:.4f} ≤ Target {target_value:.4f} (OK)"
                )

        if not drift_detected:
            logger.info("All metrics within targets. No drift detected.")
        return drift_detected

def main():
    logger.info("====== Governance Layer: PerformanceMonitor Demo ======")
    telemetry_sys = TelemetrySystem()
    monitor = PerformanceMonitor(telemetry_sys, window_size=5)

    adaptive_targets = {
        "latency_target": 0.5,
        "convergence_steps_target": 2.0,
        "false_block_rate_target": 1e-4,
        "memory_staleness_target": 3.0,
    }

    logger.info("--- Simulating 5 monitoring cycles (no drift expected) ---")
    for i in range(5):
        logger.info(f"\nCycle {i+1}:")
        monitor.record_metrics()
        monitor.check_drift(adaptive_targets)

    # Force drift
    logger.info("--- Simulating performance degradation ---")
    monitor.metric_history.setdefault("latency", deque(maxlen=5)).extend([0.7, 0.8, 0.75])
    monitor.metric_history.setdefault("convergence_steps", deque(maxlen=5)).extend([3.0, 3.0, 4.0])

    logger.info("\nRe-checking for drift with degraded metrics:")
    drift = monitor.check_drift(adaptive_targets)
    if drift:
        logger.info("CONCLUSION: Drift detected. The system should now trigger RETUNE phase.")

    logger.info("✅ performance_monitor.py executed successfully!")

if __name__ == "__main__":
    main()
