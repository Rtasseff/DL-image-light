"""Core platform utilities for SDD v4.1 compliance."""

from .config import load_config, ValidationMode, get_validation_mode
from .dependencies import (
    should_use_fallbacks,
    validate_fallback_compatibility,
    check_dependency_availability,
    get_fallback_info
)
from .effective_settings import EffectiveSettingsLogger
from .auto_tune import AutoTuner
from .trainer import SegmentationTrainer
from .experiment_identity import ExperimentIdentity, generate_run_id

__all__ = [
    "load_config",
    "ValidationMode",
    "get_validation_mode",
    "should_use_fallbacks",
    "validate_fallback_compatibility",
    "check_dependency_availability",
    "get_fallback_info",
    "EffectiveSettingsLogger",
    "AutoTuner",
    "SegmentationTrainer",
    "ExperimentIdentity",
    "generate_run_id"
]