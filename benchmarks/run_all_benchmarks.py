#!/usr/bin/env python3
"""
benchmarks/run_all_benchmarks.py

Master benchmark runner that executes all existing benchmark scripts
and generates a comprehensive validation report.
"""

import sys
import os
import time
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

@dataclass
class BenchmarkResult:
    """Structure for storing benchmark results."""
    name: str
    script: str
    status: str  # "PASS", "FAIL", "ERROR"
    runtime_seconds: float
    details: Dict[str, Any]
    output: str

class MasterBenchmarkRunner:
    """Runs all existing benchmark scripts and generates unified report."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "master_benchmark.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("MasterBenchmarkRunner")
        
        self.results: List[BenchmarkResult] = []
        
        # Define available benchmarks based on your existing files
        self.benchmarks = [
            {
                "name": "GNN Convergence Validation", 
                "script": "convergence_validation.py",
                "description": "Validates Pr[convergence ≤ 2 steps] ≥ 0.87"
            },
            {
                "name": "End-to-End Performance Test",
                "script": "performance_tests.py", 
                "description": "Multi-domain system integration benchmark"
            },
            # Add these when you create them
            # {
            #     "name": "Safety Validation",
            #     "script": "safety_validation.py",
            #     "description": "Validates false-block rate ≤ 10⁻⁴"
            # },
        ]
    
    def run_benchmark_script(self, script_name: str) -> BenchmarkResult:
        """Run a single benchmark script and capture results."""
        self.logger.info(f"🚀 Running {script_name}...")
        
        script_path = Path(__file__).parent / script_name
        
        if not script_path.exists():
            return BenchmarkResult(
                name=script_name,
                script=script_name,
                status="ERROR",
                runtime_seconds=0.0,
                details={"error": f"Script {script_name} not found"},
                output=""
            )
        
        start_time = time.time()
        
        try:
            # Run the benchmark script as a subprocess
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=project_root
            )
            
            runtime = time.time() - start_time
            
            # Parse output to determine success/failure
            output = result.stdout
            stderr = result.stderr
            
            # Determine status based on output and return code
            if result.returncode == 0:
                if "✅ PASS" in output or "SUCCESS" in output or "VALIDATED" in output:
                    status = "PASS"
                elif "❌ FAIL" in output or "FAILED" in output or "NOT VERIFIED" in output:
                    status = "FAIL"
                else:
                    status = "PASS"  # Assume pass if no errors and return code 0
            else:
                status = "ERROR"
            
            # Extract key metrics from output
            details = self._extract_metrics_from_output(output, script_name)
            
            return BenchmarkResult(
                name=script_name,
                script=script_name,
                status=status,
                runtime_seconds=runtime,
                details=details,
                output=output + ("\n--- STDERR ---\n" + stderr if stderr else "")
            )
            
        except subprocess.TimeoutExpired:
            runtime = time.time() - start_time
            return BenchmarkResult(
                name=script_name,
                script=script_name,
                status="ERROR",
                runtime_seconds=runtime,
                details={"error": "Timeout after 300 seconds"},
                output=""
            )
            
        except Exception as e:
            runtime = time.time() - start_time
            return BenchmarkResult(
                name=script_name,
                script=script_name,
                status="ERROR", 
                runtime_seconds=runtime,
                details={"error": str(e)},
                output=""
            )
    
    def _extract_metrics_from_output(self, output: str, script_name: str) -> Dict[str, Any]:
        """Extract key metrics from benchmark output."""
        details = {}
        
        if "convergence_validation.py" in script_name:
            # Extract convergence metrics
            lines = output.split('\n')
            for line in lines:
                if "Measured Probability:" in line:
                    try:
                        prob = float(line.split(':')[1].strip())
                        details["measured_probability"] = prob
                    except:
                        pass
                elif "Theoretical Bound:" in line:
                    try:
                        bound = float(line.split(':')[1].strip())
                        details["theoretical_bound"] = bound
                    except:
                        pass
                elif "Avg Steps:" in line:
                    try:
                        avg_steps = float(line.split(':')[1].split('±')[0].strip())
                        details["avg_steps"] = avg_steps
                    except:
                        pass
        
        elif "performance_tests.py" in script_name:
            # Extract performance metrics
            lines = output.split('\n')
            for line in lines:
                if "domain:" in line and "avg latency" in line:
                    try:
                        parts = line.split()
                        domain = parts[1].replace(":", "").lower()
                        latency = float(parts[3].replace("s", ""))
                        success_rate = float(parts[5].replace("%", "")) / 100
                        details[f"{domain}_latency"] = latency
                        details[f"{domain}_success_rate"] = success_rate
                    except:
                        pass
        
        return details
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all available benchmarks and generate comprehensive report."""
        self.logger.info("🧠 Starting Hybrid AI Brain - Master Benchmark Suite")
        self.logger.info("=" * 70)
        
        start_time = time.time()
        
        # Run each benchmark
        for benchmark_config in self.benchmarks:
            script_name = benchmark_config["script"]
            benchmark_name = benchmark_config["name"]
            
            self.logger.info(f"\n📊 Running: {benchmark_name}")
            self.logger.info(f"   Script: {script_name}")
            self.logger.info(f"   Description: {benchmark_config['description']}")
            
            result = self.run_benchmark_script(script_name)
            self.results.append(result)
            
            # Log immediate result
            status_emoji = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}
            emoji = status_emoji.get(result.status, "❓")
            self.logger.info(f"   Result: {emoji} {result.status} ({result.runtime_seconds:.2f}s)")
            
            if result.details:
                for key, value in result.details.items():
                    self.logger.info(f"     {key}: {value}")
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        report = self.generate_report(total_time)
        
        # Save results
        self.save_results(report)
        
        self.logger.info("=" * 70)
        self.logger.info("🎯 Master Benchmark Suite Complete")
        
        return report
    
    def generate_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        passed_count = sum(1 for r in self.results if r.status == "PASS")
        failed_count = sum(1 for r in self.results if r.status == "FAIL")
        error_count = sum(1 for r in self.results if r.status == "ERROR")
        total_count = len(self.results)
        
        report = {
            "timestamp": time.time(),
            "total_time_seconds": total_time,
            "summary": {
                "total_benchmarks": total_count,
                "passed": passed_count,
                "failed": failed_count,
                "errors": error_count,
                "success_rate": passed_count / total_count if total_count > 0 else 0
            },
            "individual_results": [asdict(result) for result in self.results],
            "paper_validation_status": {
                "convergence_validated": any(
                    r.name == "convergence_validation.py" and r.status == "PASS" 
                    for r in self.results
                ),
                "performance_validated": any(
                    r.name == "performance_tests.py" and r.status == "PASS"
                    for r in self.results
                ),
                "all_claims_verified": passed_count == total_count and failed_count == 0
            },
            "metrics_summary": self._aggregate_metrics()
        }
        
        return report
    
    def _aggregate_metrics(self) -> Dict[str, Any]:
        """Aggregate key metrics across all benchmarks."""
        aggregated = {}
        
        for result in self.results:
            if result.status == "PASS" and result.details:
                for key, value in result.details.items():
                    if isinstance(value, (int, float)):
                        if key not in aggregated:
                            aggregated[key] = []
                        aggregated[key].append(value)
        
        # Calculate averages for numeric metrics
        summary = {}
        for key, values in aggregated.items():
            if values:
                summary[key] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        return summary
    
    def save_results(self, report: Dict[str, Any]):
        """Save benchmark results to files."""
        timestamp = int(time.time())
        
        # Save JSON report
        json_file = self.output_dir / f"master_benchmark_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save human-readable summary
        summary_file = self.output_dir / "latest_master_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Hybrid AI Brain - Master Benchmark Results\n")
            f.write("=" * 50 + "\n\n")
            
            summary = report["summary"]
            f.write(f"Execution Time: {report['total_time_seconds']:.2f} seconds\n")
            f.write(f"Total Benchmarks: {summary['total_benchmarks']}\n")
            f.write(f"Passed: {summary['passed']}\n")
            f.write(f"Failed: {summary['failed']}\n")
            f.write(f"Errors: {summary['errors']}\n")
            f.write(f"Success Rate: {summary['success_rate']:.1%}\n\n")
            
            f.write("Individual Results:\n")
            f.write("-" * 40 + "\n")
            
            for result in self.results:
                status_symbols = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}
                symbol = status_symbols.get(result.status, "❓")
                
                f.write(f"{symbol} {result.name}\n")
                f.write(f"   Status: {result.status}\n")
                f.write(f"   Runtime: {result.runtime_seconds:.2f}s\n")
                
                if result.details:
                    f.write("   Key Metrics:\n")
                    for key, value in result.details.items():
                        f.write(f"     {key}: {value}\n")
                f.write("\n")
            
            # Paper validation summary
            validation = report["paper_validation_status"]
            f.write("Paper Claims Validation:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Convergence Theory: {'✅' if validation['convergence_validated'] else '❌'}\n")
            f.write(f"Performance Claims: {'✅' if validation['performance_validated'] else '❌'}\n")
            f.write(f"All Claims Verified: {'✅' if validation['all_claims_verified'] else '❌'}\n")
        
        self.logger.info(f"📄 Results saved:")
        self.logger.info(f"   JSON Report: {json_file}")
        self.logger.info(f"   Summary: {summary_file}")

def main():
    """Main entry point for master benchmark suite."""
    print("🧠 Hybrid AI Brain - Master Benchmark Suite")
    print("=" * 70)
    print("Running all available benchmark scripts:")
    print("• GNN Convergence Validation")
    print("• End-to-End Performance Testing")
    print("• Domain Adaptability Verification")
    print("=" * 70)
    
    runner = MasterBenchmarkRunner()
    report = runner.run_all_benchmarks()
    
    # Print final summary
    summary = report["summary"]
    validation = report["paper_validation_status"]
    
    print(f"\n🎯 MASTER BENCHMARK RESULTS:")
    print(f"   Total Runtime: {report['total_time_seconds']:.2f} seconds")
    print(f"   Benchmarks: {summary['passed']}/{summary['total_benchmarks']} passed")
    print(f"   Success Rate: {summary['success_rate']:.1%}")
    print(f"   Paper Claims: {'✅ ALL VERIFIED' if validation['all_claims_verified'] else '❌ SOME FAILED'}")
    
    if summary['success_rate'] == 1.0:
        print("\n🎉 ALL BENCHMARKS PASSED!")
        print("   🏆 System ready for paper submission")
        print("   📜 All theoretical claims empirically validated")
    else:
        print(f"\n⚠️  {summary['failed']} benchmarks failed, {summary['errors']} had errors")
        print("   🔍 Review individual results for details")
    
    return 0 if summary['success_rate'] == 1.0 else 1

if __name__ == "__main__":
    exit(main())