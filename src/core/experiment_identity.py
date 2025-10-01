"""
Experiment identity capture for SDD v4.1 compliance.

This module captures complete experiment context including git state,
configuration hash, and environment for full reproducibility tracking.
Implements SDD v4.1 Appendix A.3 requirements.
"""

import os
import sys
import subprocess
import hashlib
import json
import uuid
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentIdentity:
    """Capture complete experiment context per SDD v4.1."""

    @staticmethod
    def capture(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        """
        Capture all experiment identity information.

        Args:
            config: Configuration dictionary
            run_dir: Directory for this training run

        Returns:
            Dictionary with complete experiment identity
        """
        identity = {
            'run_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'git': ExperimentIdentity._capture_git_state(),
            'config_hash': ExperimentIdentity._compute_config_hash(config),
            'environment': ExperimentIdentity._capture_environment(),
            'platform': ExperimentIdentity._capture_platform(),
            'dependencies': ExperimentIdentity._capture_dependencies()
        }

        # Save immediately for audit trail
        output_path = run_dir / 'experiment_identity.json'
        with open(output_path, 'w') as f:
            json.dump(identity, f, indent=2)

        logger.info(f"Experiment identity captured: {identity['run_id']}")
        if identity['git']['dirty']:
            logger.warning(f"⚠️  Git repository has uncommitted changes!")

        return identity

    @staticmethod
    def _capture_git_state() -> Dict[str, Any]:
        """Capture git repository state."""
        try:
            # Get current commit
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()

            # Check for uncommitted changes
            status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                text=True,
                stderr=subprocess.DEVNULL
            )

            # Get branch name
            try:
                branch = subprocess.check_output(
                    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                    text=True,
                    stderr=subprocess.DEVNULL
                ).strip()
            except subprocess.CalledProcessError:
                branch = 'unknown'

            # Get diff stats if dirty
            diff_stats = None
            if status:
                try:
                    diff_stats = subprocess.check_output(
                        ['git', 'diff', '--shortstat'],
                        text=True,
                        stderr=subprocess.DEVNULL
                    ).strip()
                except subprocess.CalledProcessError:
                    diff_stats = 'unable to compute'

            # Get remote info
            remote_url = None
            try:
                remote_url = subprocess.check_output(
                    ['git', 'config', '--get', 'remote.origin.url'],
                    text=True,
                    stderr=subprocess.DEVNULL
                ).strip()
            except subprocess.CalledProcessError:
                pass

            return {
                'commit': commit,
                'branch': branch,
                'dirty': bool(status),
                'diff_stats': diff_stats,
                'remote_url': remote_url,
                'status_lines': len(status.splitlines()) if status else 0
            }

        except subprocess.CalledProcessError as e:
            logger.warning(f"Git information unavailable: {e}")
            return {
                'error': 'Not a git repository or git not available',
                'commit': None,
                'branch': None,
                'dirty': None,
                'diff_stats': None,
                'remote_url': None,
                'status_lines': None
            }

    @staticmethod
    def _compute_config_hash(config: Dict[str, Any]) -> str:
        """Compute deterministic hash of configuration."""
        # Convert config to deterministic JSON string
        config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(config_str.encode()).hexdigest()

    @staticmethod
    def _capture_environment() -> Dict[str, Any]:
        """Capture Python and system environment."""
        return {
            'python_version': sys.version,
            'python_executable': sys.executable,
            'platform': {
                'system': os.name,
                'user': os.environ.get('USER', 'unknown'),
                'pwd': str(Path.cwd()),
                'pythonpath': sys.path[:3]  # First few entries only
            },
            'environment_vars': {
                'USE_FALLBACKS': os.environ.get('USE_FALLBACKS'),
                'CONFIG_VALIDATION_MODE': os.environ.get('CONFIG_VALIDATION_MODE'),
                'CI': os.environ.get('CI'),
                'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')
            }
        }

    @staticmethod
    def _capture_platform() -> Dict[str, Any]:
        """Capture hardware and compute platform info."""
        platform_info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'cpu_count': os.cpu_count()
        }

        # Get CUDA device names if available
        if torch.cuda.is_available():
            try:
                platform_info['cuda_devices'] = [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ]
            except Exception as e:
                platform_info['cuda_devices'] = [f'Error: {e}']

        return platform_info

    @staticmethod
    def _capture_dependencies() -> Dict[str, Any]:
        """Capture key dependency versions."""
        deps = {}

        # Key packages for reproducibility
        packages = [
            'torch', 'pytorch_lightning', 'torchmetrics',
            'segmentation_models_pytorch', 'albumentations',
            'numpy', 'pillow'
        ]

        for package in packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                deps[package] = version
            except ImportError:
                deps[package] = 'not_installed'
            except Exception as e:
                deps[package] = f'error: {e}'

        return deps


def generate_run_id() -> str:
    """Generate unique run identifier."""
    return str(uuid.uuid4())


__all__ = ['ExperimentIdentity', 'generate_run_id']