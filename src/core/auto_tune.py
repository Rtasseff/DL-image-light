"""
Auto-tuning with safety gates for SDD v4.0 compliance.

This module provides opt-in auto-tuning functionality that is disabled
by default and logs all changes made to requested settings.
"""

import warnings
from typing import Dict, Any, Optional
import torch

from .effective_settings import EffectiveSettingsLogger
from .config import get_validation_mode, ValidationMode


class AutoTuner:
    """
    Auto-tune settings when explicitly requested.

    This class implements the SDD v4.0 auto-tuning policy:
    - Disabled by default
    - Opt-in only
    - Safety gates prevent use in STRICT mode
    - All changes logged via EffectiveSettingsLogger
    """

    def __init__(self, config: Dict[str, Any], logger: EffectiveSettingsLogger):
        """
        Initialize auto-tuner.

        Args:
            config: Configuration dictionary
            logger: Effective settings logger to track changes

        Raises:
            ValueError: If auto-tune is enabled in STRICT mode without force flag
        """
        self.config = config
        self.logger = logger
        self.enabled = config.get('resources', {}).get('auto_tune', False)

        # Safety check for STRICT mode
        if self.enabled:
            mode = get_validation_mode()
            if mode == ValidationMode.STRICT:
                force = config.get('resources', {}).get('force_auto_tune_in_strict', False)
                if not force:
                    raise ValueError(
                        "Auto-tune disabled in STRICT mode for safety.\n"
                        "Either:\n"
                        "1. Use PERMISSIVE mode for development\n"
                        "2. Set resources.force_auto_tune_in_strict: true"
                    )
                else:
                    warnings.warn(
                        "⚠️  Auto-tune forced in STRICT mode - "
                        "results may not be reproducible!",
                        UserWarning
                    )

    def tune_batch_size(self, requested: int) -> int:
        """
        Auto-tune batch size if enabled.

        Args:
            requested: Requested batch size

        Returns:
            Effective batch size (may be same as requested)
        """
        if not self.enabled:
            return requested

        auto_batch_config = self.config.get('resources', {}).get('auto_batch_size', {})
        if not auto_batch_config.get('enabled', False):
            return requested

        # Simple memory-based batch size suggestion
        suggested = self._suggest_batch_size_by_memory(
            requested,
            auto_batch_config.get('memory_fraction', 0.6),
            auto_batch_config.get('min_batch_size', 1),
            auto_batch_config.get('max_batch_size', 32)
        )

        if suggested != requested:
            self.logger.log_setting(
                'batch_size',
                requested=requested,
                effective=suggested,
                reason=f"Auto-tuned based on available memory"
            )

        return suggested

    def tune_precision(self, requested: int) -> int:
        """
        Auto-tune precision if enabled.

        Args:
            requested: Requested precision (16 or 32)

        Returns:
            Effective precision
        """
        if not self.enabled:
            return requested

        auto_precision_config = self.config.get('resources', {}).get('auto_precision', {})
        if not auto_precision_config.get('enabled', False):
            return requested

        # Check if mixed precision is supported
        if requested == 16:
            if not self._mixed_precision_supported():
                suggested = 32
                self.logger.log_setting(
                    'precision',
                    requested=requested,
                    effective=suggested,
                    reason="Mixed precision not supported on this platform"
                )
                return suggested

        return requested

    def _suggest_batch_size_by_memory(
        self,
        requested: int,
        memory_fraction: float,
        min_batch: int,
        max_batch: int
    ) -> int:
        """
        Suggest batch size based on available memory.

        Args:
            requested: Original requested batch size
            memory_fraction: Fraction of memory to use
            min_batch: Minimum allowed batch size
            max_batch: Maximum allowed batch size

        Returns:
            Suggested batch size
        """
        try:
            if torch.cuda.is_available():
                # Get GPU memory
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(device).total_memory
                available_memory = total_memory * memory_fraction

                # Rough estimate: each image ~50MB in float32
                estimated_per_sample = 50 * 1024 * 1024  # 50MB
                suggested = int(available_memory / estimated_per_sample)

                # Clamp to reasonable bounds
                suggested = max(min_batch, min(max_batch, suggested))

                return suggested
            else:
                # For CPU/MPS, use more conservative approach
                return min(requested, 8)

        except Exception:
            # Fallback to original if anything fails
            return requested

    def _mixed_precision_supported(self) -> bool:
        """
        Check if mixed precision is supported on current platform.

        Returns:
            True if mixed precision is supported
        """
        try:
            if torch.cuda.is_available():
                # CUDA generally supports mixed precision
                return True
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS may not support all mixed precision features
                return False
            else:
                # CPU doesn't benefit from mixed precision
                return False
        except Exception:
            return False


__all__ = ["AutoTuner"]