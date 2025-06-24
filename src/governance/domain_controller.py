#!/usr/bin/env python3
"""
src/governance/domain_controller.py

Implements the DomainController, which manages the system's operational state
(Precision, Adaptive, Exploration) based on performance metrics and manifests.
"""

import logging
from enum import Enum, auto
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger("hybrid_ai_brain.domain_controller")
logging.basicConfig(level=logging.INFO)

# --- Placeholder Dependencies ---
class ManifestManager:
    """Mock of the ManifestManager."""
    def get_manifest(self, domain_name: str) -> Optional[Dict[str, Any]]:
        logger.info(f"ManifestManager: Fetching manifest for domain '{domain_name}'.")
        mock_manifests = {
            "precision": {"domain": "precision", "parameters": {"gM": 0, "error_tolerance": 0.0, "safety_samples": 116}},
            "adaptive": {"domain": "adaptive", "parameters": {"gM": "scheduled", "error_tolerance": 0.05, "safety_samples": 59}},
            "exploration": {"domain": "exploration", "parameters": {"gM": 1, "error_tolerance": 0.20, "safety_samples": 32}},
        }
        return mock_manifests.get(domain_name.lower())

    def create_new_manifest_fork(self, base_domain: str) -> Dict[str, Any]:
        logger.info(f"ManifestManager: Forking manifest from '{base_domain}' for re-tuning.")
        return {"domain": f"retune_{base_domain}", "parameters": {}}

class PerformanceMonitor:
    """Mock of the PerformanceMonitor."""
    def check_drift(self, domain_thresholds: Dict) -> bool:
        logger.info("PerformanceMonitor: Checking for performance drift (demo: always False).")
        return False

# --- Enums for Domain/Phase ---
class DomainMode(Enum):
    PRECISION = auto()
    ADAPTIVE = auto()
    EXPLORATION = auto()

class SystemPhase(Enum):
    LEARN = auto()
    DEPLOY = auto()
    RETUNE = auto()

# --- Controller ---
class DomainController:
    """
    Orchestrates system governance through domain-adaptive manifests.
    """
    def __init__(
        self, 
        manifest_manager: ManifestManager, 
        performance_monitor: PerformanceMonitor,
        initial_domain: DomainMode = DomainMode.ADAPTIVE,
        initial_phase: SystemPhase = SystemPhase.DEPLOY
    ):
        self.manifest_manager = manifest_manager
        self.performance_monitor = performance_monitor

        self._current_domain: DomainMode = initial_domain
        self._current_phase: SystemPhase = initial_phase
        self._active_manifest: Optional[Dict[str, Any]] = None

        logger.info("DomainController initialized.")
        self.switch_domain(self._current_domain)

    @property
    def current_domain(self) -> DomainMode:
        return self._current_domain

    @property
    def current_phase(self) -> SystemPhase:
        return self._current_phase

    def switch_domain(self, new_domain: DomainMode) -> None:
        """
        Loads and activates the manifest for a new operational domain.
        """
        domain_name = new_domain.name.lower()
        logger.info(f"Switching to domain: '{domain_name.upper()}'")
        manifest = self.manifest_manager.get_manifest(domain_name)
        if manifest:
            self._active_manifest = manifest
            self._current_domain = new_domain
            logger.info(f"Successfully loaded manifest for '{domain_name}'.")
        else:
            logger.error(f"Manifest for domain '{domain_name}' not found.")
            raise ValueError(f"Manifest for domain '{domain_name}' not found.")

    def get_control_parameters(self) -> Dict[str, Any]:
        """
        Returns the parameters from the active manifest.
        """
        if not self._active_manifest:
            logger.error("No active manifest. Cannot retrieve control parameters.")
            raise RuntimeError("No active manifest.")
        logger.info("Providing control parameters from active manifest.")
        return self._active_manifest.get("parameters", {})

    def evaluate_system_state(self) -> None:
        """
        State machine logic for governance: transitions phases based on performance.
        """
        logger.info(f"Evaluating system state. Current phase: {self._current_phase.name}")
        if self._current_phase == SystemPhase.DEPLOY:
            domain_thresholds = self._active_manifest.get("performance_targets", {}) if self._active_manifest else {}
            if self.performance_monitor.check_drift(domain_thresholds):
                logger.info("Performance drift detected. Transitioning to RETUNE phase.")
                self._current_phase = SystemPhase.RETUNE
                self.manifest_manager.create_new_manifest_fork(self._current_domain.name.lower())
            else:
                logger.info("Performance stable. Remaining in DEPLOY phase.")
        elif self._current_phase == SystemPhase.RETUNE:
            logger.info("In RETUNE phase. Awaiting validation of new manifest fork.")
        elif self._current_phase == SystemPhase.LEARN:
            logger.info("In LEARN phase. Training models and establishing baseline.")

# --- Demo Block ---
def main():
    logger.info("====== Governance Layer: DomainController Demo ======")
    manifest_mgr = ManifestManager()
    perf_monitor = PerformanceMonitor()
    controller = DomainController(manifest_mgr, perf_monitor)

    # Get control parameters for current domain
    params = controller.get_control_parameters()
    print(f"Initial (ADAPTIVE) Parameters: gM='{params.get('gM')}', samples={params.get('safety_samples')}")

    # Switch domain
    controller.switch_domain(DomainMode.PRECISION)
    precision_params = controller.get_control_parameters()
    print(f"Switched to PRECISION. Parameters: gM={precision_params.get('gM')}, samples={precision_params.get('safety_samples')}")

    # Evaluate system state (should remain in DEPLOY)
    controller.evaluate_system_state()
    logger.info("✅ domain_controller.py executed successfully!")

if __name__ == "__main__":
    main()
