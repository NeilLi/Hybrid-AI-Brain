#!/usr/bin/env python3
"""
src/safety/safety_monitor.py

Integrates RiskAssessor and GraphMask for a complete safety pipeline.
"""

import logging
from typing import Dict, Any, Optional

from .risk_assessor import RiskAssessor
from .graph_mask import GraphMask

logger = logging.getLogger("hybrid_ai_brain.safety.safety_monitor")
logging.basicConfig(level=logging.INFO)

class SafetyMonitor:
    """
    Orchestrates the safety-checking process, combining heuristic risk assessment
    with model-based GraphMask validation.
    """
    def __init__(self, risk_assessor: RiskAssessor, graph_mask: GraphMask):
        self.risk_assessor = risk_assessor
        self.graph_mask = graph_mask
        logger.info("SafetyMonitor initialized, integrating RiskAssessor and GraphMask.")

    def adjudicate_edge(
        self,
        edge_features: Dict[str, Any],
        domain_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Runs the edge through risk and mask validation.
        Args:
            edge_features: Dict of features for the proposed action/edge.
            domain_params: Optional dict with e.g. tau_safe, safety_samples.
        Returns:
            True if approved, False if blocked.
        """
        logger.info(f"Adjudicating edge with features: {edge_features}")
        domain_params = domain_params or {}

        # 1. Heuristic risk score
        heuristic_risk_score = self.risk_assessor.assess(edge_features)
        mask_input = dict(edge_features)
        mask_input['heuristic_risk'] = heuristic_risk_score

        # 2. Rigorous model/statistical validation
        is_safe = self.graph_mask.is_edge_safe(
            mask_input,
            safety_samples=domain_params.get("safety_samples", 59),
            tau_safe=domain_params.get("tau_safe_threshold", 0.7)
        )

        if not is_safe:
            logger.warning("ACTION BLOCKED. Logging safety violation for audit.")
        else:
            logger.info("Action approved by full safety pipeline.")

        return is_safe

def main():
    logger.info("====== Safety Layer: SafetyMonitor Demo ======")
    assessor = RiskAssessor()
    mask = GraphMask()
    monitor = SafetyMonitor(assessor, mask)

    # High-risk action scenario
    high_risk_action_features = {
        "action": "delete",
        "data_access": "financial",
        "user_role": "analyst"
    }

    print("\n--- Adjudicating high-risk action with ADAPTIVE domain settings ---")
    safe1 = monitor.adjudicate_edge(high_risk_action_features)
    print(f"ADAPTIVE domain result: {'APPROVED' if safe1 else 'BLOCKED'}")

    print("\n--- Adjudicating high-risk action with PRECISION domain settings ---")
    precision_params = {"safety_samples": 116, "tau_safe_threshold": 0.8}
    safe2 = monitor.adjudicate_edge(high_risk_action_features, domain_params=precision_params)
    print(f"PRECISION domain result: {'APPROVED' if safe2 else 'BLOCKED'}")

    logger.info("✅ safety_monitor.py executed successfully!\n")

if __name__ == "__main__":
    main()
