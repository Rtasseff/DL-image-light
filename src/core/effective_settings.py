"""
Effective Settings Logging for SDD v4.1 compliance.

This module tracks and logs actual vs requested settings to ensure
transparency in what the system actually does vs what was requested.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime


class EffectiveSettingsLogger:
    """
    Track and log actual vs requested settings.

    This class implements the SDD v4.1 requirement that all automatic
    behaviors must be logged and deviations from requested settings
    must be explicitly tracked.
    """

    def __init__(self, run_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the logger.

        Args:
            run_dir: Directory where this run's outputs are stored (default: ./runs)
        """
        self.run_dir = Path(run_dir) if run_dir else Path("./runs")
        self.requested = {}
        self.effective = {}
        self.changes = []

    def log_setting(
        self,
        key: str,
        requested: Any,
        effective: Any,
        reason: str = None
    ):
        """
        Log a setting that may have changed from requested.

        Args:
            key: Setting name (e.g., 'batch_size', 'precision')
            requested: What was originally requested
            effective: What is actually being used
            reason: Why the change was made (if any)
        """
        self.requested[key] = requested
        self.effective[key] = effective

        if requested != effective:
            change = {
                'setting': key,
                'requested': requested,
                'effective': effective,
                'reason': reason or 'Unknown',
                'timestamp': datetime.now().isoformat()
            }
            self.changes.append(change)

            # Warn user about change
            warnings.warn(
                f"Setting changed: {key} = {effective} "
                f"(requested: {requested}). Reason: {reason}",
                UserWarning
            )

    def save(self):
        """Save effective settings to file."""
        output = {
            'requested_settings': self.requested,
            'effective_settings': self.effective,
            'changes': self.changes,
            'change_count': len(self.changes),
            'fallbacks_used': self._detect_fallbacks(),
            'platform_optimizations': self._detect_optimizations(),
            'generation_time': datetime.now().isoformat(),
            'run_directory': str(self.run_dir)
        }

        output_path = self.run_dir / 'effective_settings.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        if self.changes:
            print(f"\n⚠️  {len(self.changes)} settings changed from requested values.")
            print(f"   See {output_path} for details.\n")

    def _detect_fallbacks(self) -> List[str]:
        """
        Detect which fallback implementations are in use.

        Returns:
            List of fallback implementations currently active
        """
        fallbacks = []

        try:
            from ..dependencies import should_use_fallbacks
            if should_use_fallbacks():
                fallbacks.append('fallback_mode_enabled')
        except ImportError:
            pass

        # Check specific fallback implementations
        try:
            from ..metrics import get_metric
            # Try to detect if we're using simple metrics
            metric_fn = get_metric('dice')
            if hasattr(metric_fn, '__module__') and 'simple' in metric_fn.__module__:
                fallbacks.append('simple_metrics')
        except Exception:
            pass

        return fallbacks

    def _detect_optimizations(self) -> List[str]:
        """
        Detect platform optimizations applied.

        Returns:
            List of platform optimizations in use
        """
        import torch
        optimizations = []

        if torch.cuda.is_available():
            optimizations.append('cuda_enabled')
            optimizations.append(f'cuda_device_count_{torch.cuda.device_count()}')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            optimizations.append('mps_enabled')
        else:
            optimizations.append('cpu_only')

        # Check for mixed precision
        try:
            if hasattr(torch, 'autocast'):
                optimizations.append('autocast_available')
        except Exception:
            pass

        return optimizations

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of effective settings.

        Returns:
            Dictionary with summary information
        """
        return {
            'total_settings': len(self.requested),
            'changed_settings': len(self.changes),
            'fallbacks_used': len(self._detect_fallbacks()) > 0,
            'platform': self._detect_optimizations(),
            'changes': self.changes
        }


__all__ = ["EffectiveSettingsLogger"]