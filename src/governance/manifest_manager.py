#!/usr/bin/env python3
"""
src/governance/manifest_manager.py

Handles loading, validation, and management of domain-specific YAML manifests.
"""

import yaml
import logging
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger("hybrid_ai_brain.manifest_manager")
logging.basicConfig(level=logging.INFO)

class ManifestManager:
    """
    Loads and validates domain manifests from the config directory.
    Manifests are YAML files for each operational domain.
    """
    def __init__(self, config_dir: str = "configs/domains"):
        self.config_path = Path(config_dir)
        if not self.config_path.is_dir():
            raise FileNotFoundError(
                f"Configuration directory not found at '{self.config_path.resolve()}'. "
                "Ensure the project structure is correct."
            )
        logger.info(f"ManifestManager initialized. Watching directory: '{self.config_path}'.")

    def list_domains(self) -> List[str]:
        """Lists all available manifest YAMLs (domain names)."""
        return [
            f.stem for f in self.config_path.glob("*.yaml")
            if f.is_file()
        ]

    def _validate_manifest(self, manifest_data: Dict[str, Any], domain_name: str) -> bool:
        """
        Validates a loaded manifest against the expected schema.
        """
        required_keys = ["domain", "description", "parameters", "coordinator_config"]
        missing_keys = [key for key in required_keys if key not in manifest_data]
        if missing_keys:
            logger.error(f"Manifest '{domain_name}' missing keys: {missing_keys}")
            return False

        # Minimal parameter checks
        required_params = ["gM", "error_tolerance", "safety_samples", "tau_safe_threshold"]
        params = manifest_data.get("parameters", {})
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            logger.error(f"Manifest '{domain_name}' missing parameters: {missing_params}")
            return False

        logger.info(f"Manifest '{domain_name}' passed validation.")
        return True

    def get_manifest(self, domain_name: str) -> Optional[Dict[str, Any]]:
        """
        Loads and validates a domain manifest.
        """
        manifest_file = self.config_path / f"{domain_name.lower()}.yaml"
        logger.info(f"Attempting to load manifest: '{manifest_file}'")

        if not manifest_file.exists():
            logger.warning(f"File not found for domain '{domain_name}'.")
            return None

        try:
            with open(manifest_file, 'r') as f:
                manifest_data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Could not parse YAML for '{domain_name}': {e}")
            return None

        if self._validate_manifest(manifest_data, domain_name):
            return manifest_data
        else:
            logger.warning(f"Manifest '{domain_name}' failed validation.")
            return None

    def create_new_manifest_fork(self, base_domain: str) -> Optional[Dict[str, Any]]:
        """
        Creates a deep copy of an existing manifest for re-tuning.
        """
        logger.info(f"Forking new manifest from base '{base_domain}' for re-tuning.")
        base_manifest = self.get_manifest(base_domain)
        if not base_manifest:
            logger.error(f"Cannot fork non-existent base domain '{base_domain}'.")
            return None

        forked_manifest = copy.deepcopy(base_manifest)
        forked_manifest['domain'] = f"retune_{base_domain}"
        forked_manifest['description'] = f"Re-tuning fork of {base_domain}."

        # Optionally, the fork could be saved to disk for further editing.
        # new_filename = self.config_path / f"{forked_manifest['domain']}.yaml"
        # with open(new_filename, 'w') as f:
        #     yaml.dump(forked_manifest, f, default_flow_style=False, indent=2)

        logger.info(f"Created new manifest fork named '{forked_manifest['domain']}'.")
        return forked_manifest

def main():
    logger.info("====== Governance Layer: ManifestManager Demo ======")
    try:
        manifest_mgr = ManifestManager()
        domains = manifest_mgr.list_domains()
        logger.info(f"Available domains: {domains}")

        # 1. Load an existing manifest (e.g., 'adaptive')
        if 'adaptive' in domains:
            logger.info("--- Loading ADAPTIVE manifest ---")
            adaptive_manifest = manifest_mgr.get_manifest("adaptive")
            if adaptive_manifest:
                print("Successfully loaded ADAPTIVE manifest:")
                print(yaml.dump(adaptive_manifest, indent=2))
            else:
                print("Failed to load or validate 'adaptive' manifest.")

        # 2. Attempt to load a non-existent manifest
        logger.info("--- Loading FAKE manifest ---")
        fake_manifest = manifest_mgr.get_manifest("fake_domain")
        if not fake_manifest:
            print("Correctly handled non-existent manifest.")

        # 3. Fork the 'precision' manifest for re-tuning
        if 'precision' in domains:
            logger.info("--- Forking PRECISION manifest for re-tuning ---")
            forked = manifest_mgr.create_new_manifest_fork("precision")
            if forked:
                print("Successfully forked PRECISION manifest:")
                print(yaml.dump(forked, indent=2))
            else:
                print("Failed to fork 'precision' manifest.")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please run the initial project setup script to create the config directory.")

    logger.info("✅ manifest_manager.py executed successfully!")

if __name__ == "__main__":
    main()
