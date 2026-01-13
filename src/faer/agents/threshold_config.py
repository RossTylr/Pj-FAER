"""Site-configurable threshold management.

Allows hospitals to customize alert thresholds to their specific:
- Patient population characteristics
- Historical performance baselines
- Local clinical protocols
- Regional regulatory requirements

Configuration can be loaded from:
1. YAML/JSON files in a config directory
2. Environment variables (for deployment)
3. Streamlit session state (for interactive adjustment)

Example usage:
    from faer.agents.threshold_config import load_site_config

    # Load custom thresholds
    config = load_site_config(Path("config/thresholds/my_hospital.yaml"))
    thresholds = config.build_thresholds()

    # Create agent with custom thresholds
    agent = HeuristicShadowAgent(thresholds=thresholds)
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .interface import InsightCategory, Severity
from .shadow import ClinicalThreshold, NHS_THRESHOLDS


@dataclass
class SiteThresholdConfig:
    """Site-specific threshold configuration.

    Attributes:
        site_id: Unique identifier (e.g., "hospital_abc", "trauma_center_1")
        site_name: Human-readable site name
        description: Optional description of this configuration
        base_thresholds: Which base set to extend ("NHS", "NONE")
        overrides: Dict of metric name -> threshold value overrides
        custom_thresholds: Additional custom ClinicalThreshold rules as dicts
    """

    site_id: str
    site_name: str
    description: str = ""
    base_thresholds: str = "NHS"  # "NHS" or "NONE"
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    custom_thresholds: list[dict[str, Any]] = field(default_factory=list)

    def build_thresholds(self) -> list[ClinicalThreshold]:
        """Build final threshold list from config.

        Starts with base thresholds (NHS or none), applies overrides,
        then adds custom thresholds.

        Returns:
            List of ClinicalThreshold objects ready for agent use
        """
        # Start with base thresholds
        if self.base_thresholds.upper() == "NHS":
            # Create mutable copies of NHS thresholds
            thresholds = [self._threshold_to_dict(t) for t in NHS_THRESHOLDS]
        else:
            thresholds = []

        # Apply overrides to matching thresholds
        for metric, override_values in self.overrides.items():
            for i, threshold_dict in enumerate(thresholds):
                if threshold_dict["metric"] == metric:
                    threshold_dict.update(override_values)
                    thresholds[i] = threshold_dict
                    break

        # Add custom thresholds
        for custom in self.custom_thresholds:
            thresholds.append(custom.copy())

        # Convert all dicts back to ClinicalThreshold objects
        result = []
        for t in thresholds:
            result.append(self._dict_to_threshold(t))

        return result

    @staticmethod
    def _threshold_to_dict(threshold: ClinicalThreshold) -> dict[str, Any]:
        """Convert ClinicalThreshold to dict for modification."""
        return {
            "metric": threshold.metric,
            "threshold": threshold.threshold,
            "operator": threshold.operator,
            "severity": threshold.severity.value,
            "category": threshold.category.value,
            "title": threshold.title,
            "message_template": threshold.message_template,
            "recommendation": threshold.recommendation,
            "soft_margin": threshold.soft_margin,
            "severity_when_uncertain": (
                threshold.severity_when_uncertain.value
                if threshold.severity_when_uncertain
                else None
            ),
            "uncertainty_message_template": threshold.uncertainty_message_template,
        }

    @staticmethod
    def _dict_to_threshold(d: dict[str, Any]) -> ClinicalThreshold:
        """Convert dict back to ClinicalThreshold."""
        # Handle severity enum
        severity = d["severity"]
        if isinstance(severity, str):
            severity = Severity[severity.upper()]

        # Handle category enum
        category = d["category"]
        if isinstance(category, str):
            category = InsightCategory[category.upper()]

        # Handle optional severity_when_uncertain
        severity_when_uncertain = d.get("severity_when_uncertain")
        if severity_when_uncertain and isinstance(severity_when_uncertain, str):
            severity_when_uncertain = Severity[severity_when_uncertain.upper()]

        return ClinicalThreshold(
            metric=d["metric"],
            threshold=d["threshold"],
            operator=d["operator"],
            severity=severity,
            category=category,
            title=d["title"],
            message_template=d["message_template"],
            recommendation=d["recommendation"],
            soft_margin=d.get("soft_margin", 0.0),
            severity_when_uncertain=severity_when_uncertain,
            uncertainty_message_template=d.get("uncertainty_message_template", ""),
        )


def load_site_config(config_path: Path) -> SiteThresholdConfig:
    """Load site configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file (.yaml, .yml, or .json)

    Returns:
        SiteThresholdConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is not supported
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        if config_path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                data = yaml.safe_load(f)
            except ImportError:
                raise ImportError(
                    "PyYAML is required to load YAML config files. "
                    "Install with: pip install pyyaml"
                )
        elif config_path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(
                f"Unsupported config format: {config_path.suffix}. "
                "Use .yaml, .yml, or .json"
            )

    return SiteThresholdConfig(**data)


def save_site_config(config: SiteThresholdConfig, config_path: Path) -> None:
    """Save site configuration to YAML or JSON file.

    Args:
        config: Configuration to save
        config_path: Path to save to (.yaml, .yml, or .json)

    Raises:
        ValueError: If file format is not supported
    """
    data = {
        "site_id": config.site_id,
        "site_name": config.site_name,
        "description": config.description,
        "base_thresholds": config.base_thresholds,
        "overrides": config.overrides,
        "custom_thresholds": config.custom_thresholds,
    }

    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        if config_path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                raise ImportError(
                    "PyYAML is required to save YAML config files. "
                    "Install with: pip install pyyaml"
                )
        elif config_path.suffix == ".json":
            json.dump(data, f, indent=2)
        else:
            raise ValueError(
                f"Unsupported config format: {config_path.suffix}. "
                "Use .yaml, .yml, or .json"
            )


def get_default_config_dir() -> Path:
    """Get default configuration directory.

    Checks in order:
    1. FAER_CONFIG_DIR environment variable
    2. ./config/thresholds directory
    3. Package data directory (fallback)

    Returns:
        Path to configuration directory
    """
    # Check environment variable first
    if env_dir := os.environ.get("FAER_CONFIG_DIR"):
        return Path(env_dir)

    # Check current working directory
    cwd_config = Path.cwd() / "config" / "thresholds"
    if cwd_config.exists():
        return cwd_config

    # Fallback to package directory
    return Path(__file__).parent / "default_config"


def list_available_configs(config_dir: Path | None = None) -> list[Path]:
    """List all available configuration files.

    Args:
        config_dir: Directory to search. Uses default if None.

    Returns:
        List of paths to configuration files
    """
    if config_dir is None:
        config_dir = get_default_config_dir()

    if not config_dir.exists():
        return []

    configs = []
    for suffix in (".yaml", ".yml", ".json"):
        configs.extend(config_dir.glob(f"*{suffix}"))

    return sorted(configs)
