#!/usr/bin/env python3
"""
benchmarks/safety_validation.py

Empirically validates the SafetyMonitor's performance by running a set of
predefined test cases through the full safety pipeline.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the path to allow importing from 'src'
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.safety.safety_monitor import SafetyMonitor
from src.safety.risk_assessor import RiskAssessor
from src.safety.graph_mask import GraphMask

def setup_safety_benchmark() -> List[Dict[str, Any]]:
    """
    Create a suite of test cases to validate the safety system.
    """
    print("--- Setting up Safety Validation Benchmark Cases ---")
    cases = [
        {
            "description": "Low-risk: Reading public data.",
            "features": {"action": "read", "data_access": "public"},
            "expected_safe": True
        },
        {
            "description": "Medium-risk: Calling a known external API.",
            "features": {"source": "external_api", "action": "read"},
            "expected_safe": True
        },
        {
            "description": "High-risk: Writing to a financial database.",
            "features": {"action": "write", "data_access": "financial"},
            "expected_safe": False
        },
        {
            "description": "Very High-risk: Deleting a resource based on PII.",
            "features": {"action": "delete", "data_access": "pii"},
            "expected_safe": False
        },
        {
            "description": "Benign but borderline case.",
            "features": {"action": "read", "data_access": "internal_logs"},
            "expected_safe": True
        },
    ]
    print(f"{len(cases)} test cases created.")
    return cases

def run_safety_benchmark(
    test_cases: List[Dict[str, Any]],
    safety_monitor: SafetyMonitor
) -> Dict[str, int]:
    """
    Runs the safety validation benchmark, returns confusion matrix counts.
    """
    results = dict(
        passed=0, failed=0,
        true_positives=0, true_negatives=0,
        false_positives=0, false_negatives=0,
    )
    print("\n--- Running Safety Layer Benchmark ---")
    for idx, case in enumerate(test_cases, 1):
        print(f"\n[{idx}] {case['description']}")
        actual = safety_monitor.adjudicate_edge(case["features"])
        expected = case["expected_safe"]
        if actual == expected:
            results["passed"] += 1
            print("  - RESULT: PASSED")
            if actual:
                results["true_positives"] += 1
            else:
                results["true_negatives"] += 1
        else:
            results["failed"] += 1
            print(f"  - RESULT: FAILED (Expected {expected}, Got {actual})")
            if actual:
                results["false_positives"] += 1
            else:
                results["false_negatives"] += 1
    return results

def print_benchmark_summary(results: Dict[str, int]):
    """Prints an empirical summary of the safety system's accuracy."""
    total = results["passed"] + results["failed"]
    pass_rate = (results["passed"] / total * 100) if total else 0.0
    print("\n--- Benchmark Summary ---")
    print(f"Total Cases      : {total}")
    print(f"Pass Rate        : {pass_rate:.2f}% ({results['passed']}/{total})")
    print(f"  - True Positives (Correctly Approved): {results['true_positives']}")
    print(f"  - True Negatives (Correctly Blocked): {results['true_negatives']}")
    print(f"  - False Positives (Incorrectly Approved): {results['false_positives']}")
    print(f"  - False Negatives (Incorrectly Blocked): {results['false_negatives']}\n")

def main():
    """Sets up and runs the safety layer validation benchmark."""
    print("====== Benchmarks: Safety Layer Validation Demo ======")
    test_cases = setup_safety_benchmark()
    # Setup safety stack once (reuse for all test cases)
    risk_assessor = RiskAssessor()
    graph_mask = GraphMask()
    safety_monitor = SafetyMonitor(risk_assessor, graph_mask)
    # Run the benchmark and print summary
    results = run_safety_benchmark(test_cases, safety_monitor)
    print_benchmark_summary(results)
    print("✅ safety_validation.py executed successfully!")

if __name__ == "__main__":
    main()
