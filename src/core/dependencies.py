"""
Dependency management and fallback control.

This module implements the SDD v4.0 fallback policy with explicit control
over when fallback implementations are used.
"""

import os
import warnings
from typing import Dict, Any


def should_use_fallbacks() -> bool:
    """
    Determine if fallback implementations should be used.

    Returns:
        bool: True if fallbacks should be used, False otherwise
    """
    # Check explicit environment variable
    if os.environ.get('USE_FALLBACKS') == 'true':
        warnings.warn(
            "⚠️  Using fallback implementations. "
            "This should ONLY be used for testing or local development. "
            "Never deploy with fallbacks enabled.",
            UserWarning
        )
        return True

    # Auto-enable for specific test contexts
    if os.environ.get('PYTEST_CURRENT_TEST'):
        test_file = os.environ.get('PYTEST_CURRENT_TEST', '')
        if 'unit' in test_file or 'minimal' in test_file:
            return True

    return False


def validate_fallback_compatibility():
    """
    Ensure fallbacks aren't used with STRICT validation.

    Raises:
        RuntimeError: If fallbacks are used with STRICT validation
    """
    if should_use_fallbacks():
        from .config import get_validation_mode, ValidationMode

        mode = get_validation_mode()
        if mode == ValidationMode.STRICT:
            raise RuntimeError(
                "❌ Cannot use fallbacks with STRICT validation.\n"
                "   Either:\n"
                "   1. Disable fallbacks: export USE_FALLBACKS=false\n"
                "   2. Use PERMISSIVE mode: export CONFIG_VALIDATION_MODE=PERMISSIVE"
            )


def check_dependency_availability() -> Dict[str, bool]:
    """
    Check availability of key dependencies.

    Returns:
        Dict mapping dependency names to availability status
    """
    dependencies = {}

    # Check ML dependencies
    try:
        import torchmetrics
        dependencies['torchmetrics'] = True
    except ImportError:
        dependencies['torchmetrics'] = False

    try:
        import albumentations
        dependencies['albumentations'] = True
    except ImportError:
        dependencies['albumentations'] = False

    try:
        import segmentation_models_pytorch
        dependencies['segmentation_models_pytorch'] = True
    except ImportError:
        dependencies['segmentation_models_pytorch'] = False

    try:
        import pytorch_lightning
        dependencies['pytorch_lightning'] = True
    except ImportError:
        dependencies['pytorch_lightning'] = False

    return dependencies


def get_fallback_info() -> Dict[str, Any]:
    """
    Get information about current fallback usage.

    Returns:
        Dictionary with fallback status and details
    """
    fallbacks_enabled = should_use_fallbacks()
    dependencies = check_dependency_availability()

    missing_deps = [name for name, available in dependencies.items() if not available]

    return {
        'fallbacks_enabled': fallbacks_enabled,
        'missing_dependencies': missing_deps,
        'dependency_status': dependencies,
        'golden_path_ready': len(missing_deps) == 0
    }


__all__ = [
    "should_use_fallbacks",
    "validate_fallback_compatibility",
    "check_dependency_availability",
    "get_fallback_info"
]