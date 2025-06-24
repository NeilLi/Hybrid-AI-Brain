#!/usr/bin/env python3
"""
src/safety/risk_assessor.py

Standalone module for assessing the inherent risk of a task or an edge in the execution graph.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("hybrid_ai_brain.safety.risk_assessor")
logging.basicConfig(level=logging.INFO)

class RiskAssessor:
    """
    Calculates a heuristic risk score (ρ_ij) for a given task or edge.
    Rule-based by default, but easily extensible to a model.
    """
    def __init__(self, risk_rules: Optional[Dict[str, float]] = None):
        """
        Args:
            risk_rules: Optionally provide a dictionary of custom risk scores.
        """
        self.risk_rules = risk_rules or {
            "data_access:pii": 0.9,
            "data_access:financial": 0.8,
            "action:write": 0.7,
            "action:delete": 0.95,
            "source:external_api": 0.5,
            "default": 0.1,
        }
        logger.info("RiskAssessor initialized with rules: %s", self.risk_rules)

    def assess(self, features: Dict[str, Any]) -> float:
        """
        Assesses the risk score ρ ∈ [0,1] for given features.
        Args:
            features: Feature dictionary describing the edge/task.
        Returns:
            Risk score.
        """
        logger.info(f"Assessing features: {features}")
        max_risk = self.risk_rules.get("default", 0.1)

        # Evaluate all possible rule matches for maximum risk
        for feature_type in ["data_access", "action", "source"]:
            val = features.get(feature_type)
            if val:
                rule_key = f"{feature_type}:{val}"
                risk_val = self.risk_rules.get(rule_key)
                if risk_val is not None:
                    logger.debug(f"  Matched rule '{rule_key}' -> risk {risk_val}")
                    max_risk = max(max_risk, risk_val)

        logger.info(f"Final assessed risk score ρ = {max_risk:.2f}")
        return max_risk

def main():
    logger.info("====== Safety Layer: RiskAssessor Demo ======")
    assessor = RiskAssessor()

    # Low-risk action
    logger.info("--- Assessing a low-risk 'read' action on public data ---")
    low_risk_features = {"action": "read", "data_access": "public"}
    risk_score1 = assessor.assess(low_risk_features)
    print(f"Low-risk score: {risk_score1:.2f}")

    # High-risk action
    logger.info("--- Assessing a high-risk 'write' action on PII data ---")
    high_risk_features = {"action": "write", "data_access": "pii"}
    risk_score2 = assessor.assess(high_risk_features)
    print(f"High-risk score: {risk_score2:.2f}")

    # Action from external source
    logger.info("--- Assessing an action from an external API ---")
    external_features = {"source": "external_api", "action": "read"}
    risk_score3 = assessor.assess(external_features)
    print(f"External API risk score: {risk_score3:.2f}")

    logger.info("✅ risk_assessor.py executed successfully!")

if __name__ == "__main__":
    main()
