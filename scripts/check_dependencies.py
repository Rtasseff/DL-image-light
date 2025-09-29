#!/usr/bin/env python3
"""
Comprehensive dependency checker for SDD v4.0 compliance.

This tool verifies Golden Path dependencies and provides recommendations
for fixing dependency issues. Implements SDD v4.0 diagnostic requirements.
"""

import os
import sys
import json
import platform
import importlib.util
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime


def check_golden_path() -> Tuple[bool, Dict[str, str]]:
    """Check if golden path dependencies are satisfied."""
    # Golden Path dependencies from requirements/ml.txt
    required = {
        'torch': '2.1.0',
        'pytorch_lightning': '2.1.2',
        'segmentation_models_pytorch': '0.3.3',
        'torchmetrics': '1.2.0',
        'albumentations': '1.3.1',
        'timm': '0.9.12',
        'numpy': '1.24.3',
        'pillow': '10.1.0',
        'pyyaml': '6.0.1',
        'pydantic': '2.5.0',
        'tqdm': '4.66.1'
    }

    all_present = True
    results = {}

    print("🏆 Golden Path Dependencies:")
    print("-" * 50)

    for package, expected_version in required.items():
        actual_version = get_package_version(package)
        if actual_version == expected_version:
            print(f"  ✅ {package:30} {actual_version}")
            results[package] = {'status': 'exact', 'version': actual_version}
        elif actual_version:
            print(f"  ⚠️  {package:30} {actual_version} (expected {expected_version})")
            results[package] = {'status': 'different', 'version': actual_version, 'expected': expected_version}
            all_present = False
        else:
            print(f"  ❌ {package:30} Not installed")
            results[package] = {'status': 'missing', 'expected': expected_version}
            all_present = False

    return all_present, results


def check_base_dependencies() -> Tuple[bool, Dict[str, str]]:
    """Check if base (fallback) dependencies are satisfied."""
    # Base dependencies from requirements/base.txt
    required = [
        'torch', 'numpy', 'pillow', 'pyyaml', 'pydantic'
    ]

    all_present = True
    results = {}

    print("\n🔧 Base Dependencies (Fallback Support):")
    print("-" * 50)

    for package in required:
        actual_version = get_package_version(package)
        if actual_version:
            print(f"  ✅ {package:30} {actual_version}")
            results[package] = {'status': 'present', 'version': actual_version}
        else:
            print(f"  ❌ {package:30} Not installed")
            results[package] = {'status': 'missing'}
            all_present = False

    return all_present, results


def get_package_version(package: str) -> str:
    """Get installed package version."""
    # Handle special cases for import names
    import_name = package
    if package == 'pytorch_lightning':
        import_name = 'pytorch_lightning'  # Try original first

    try:
        spec = importlib.util.find_spec(import_name)
        if spec:
            module = importlib.import_module(import_name)
            return getattr(module, '__version__', 'unknown')
    except ImportError:
        # Try alternative import names for pytorch_lightning
        if package == 'pytorch_lightning':
            try:
                module = importlib.import_module('lightning')
                return getattr(module, '__version__', 'unknown')
            except ImportError:
                pass
    except Exception:
        pass
    return None


def check_platform() -> Dict[str, any]:
    """Check platform and compute capabilities."""
    print("\n💻 Platform Information:")
    print("-" * 50)

    try:
        import torch
        torch_available = True
        torch_version = torch.__version__
    except ImportError:
        torch_available = False
        torch_version = None

    platform_info = {
        'os': platform.system(),
        'os_release': platform.release(),
        'python_version': platform.python_version(),
        'python_executable': sys.executable,
        'torch_version': torch_version,
        'cpu_count': os.cpu_count()
    }

    print(f"  OS:          {platform_info['os']} {platform_info['os_release']}")
    print(f"  Python:      {platform_info['python_version']}")
    print(f"  Python Path: {platform_info['python_executable']}")
    print(f"  CPU Count:   {platform_info['cpu_count']}")

    if torch_available:
        platform_info.update({
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda,
            'cuda_device_count': torch.cuda.device_count(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        })

        if torch.cuda.is_available():
            print(f"  CUDA:        ✅ {torch.version.cuda}")
            print(f"  GPU Count:   {torch.cuda.device_count()}")
            try:
                gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                for i, name in enumerate(gpu_names):
                    print(f"  GPU {i}:       {name}")
                platform_info['gpu_names'] = gpu_names
            except Exception:
                pass
        else:
            print(f"  CUDA:        ❌ Not available")

        if platform_info['mps_available']:
            print(f"  MPS:         ✅ Available")
        else:
            print(f"  MPS:         ❌ Not available")

    else:
        print(f"  PyTorch:     ❌ Not installed")
        platform_info.update({
            'cuda_available': False,
            'cuda_version': None,
            'cuda_device_count': 0,
            'mps_available': False
        })

    return platform_info


def check_environment() -> Dict[str, str]:
    """Check SDD v4.0 relevant environment variables."""
    print("\n🌍 Environment Configuration:")
    print("-" * 50)

    env_vars = {
        'USE_FALLBACKS': os.environ.get('USE_FALLBACKS'),
        'CONFIG_VALIDATION_MODE': os.environ.get('CONFIG_VALIDATION_MODE'),
        'CI': os.environ.get('CI'),
        'USER': os.environ.get('USER'),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')
    }

    for var, value in env_vars.items():
        if value:
            print(f"  {var:25} = {value}")
        else:
            print(f"  {var:25} = (not set)")

    return env_vars


def provide_recommendations(golden_ok: bool, base_ok: bool, platform_info: Dict) -> List[str]:
    """Provide recommendations based on findings."""
    recommendations = []

    print("\n💡 Recommendations:")
    print("-" * 50)

    if golden_ok:
        print("  ✅ Golden path satisfied! You're ready for production.")
        recommendations.append("Golden Path satisfied - production ready")
    else:
        print("  ⚠️  Golden path not satisfied.")
        print("     For production: pip install -r requirements/ml.txt")
        recommendations.append("Install Golden Path: pip install -r requirements/ml.txt")

        if base_ok:
            print("     For development: export USE_FALLBACKS=true")
            print("                      export CONFIG_VALIDATION_MODE=PERMISSIVE")
            recommendations.append("Development mode available with fallbacks")
        else:
            print("     Minimum requirements: pip install -r requirements/base.txt")
            recommendations.append("Install base requirements first")

    # Platform-specific recommendations
    if not platform_info.get('cuda_available', False) and not platform_info.get('mps_available', False):
        print("  ⚠️  No GPU acceleration available - training will be slow")
        recommendations.append("Consider GPU access for better performance")

    # Environment recommendations
    use_fallbacks = os.environ.get('USE_FALLBACKS')
    validation_mode = os.environ.get('CONFIG_VALIDATION_MODE')

    if use_fallbacks == 'true':
        print("  ⚠️  Fallbacks enabled - not suitable for production")
        recommendations.append("Disable fallbacks for production: unset USE_FALLBACKS")

    if validation_mode and validation_mode.lower() != 'strict':
        print(f"  ⚠️  Validation mode is {validation_mode} - use STRICT for production")
        recommendations.append("Use STRICT validation for production")

    return recommendations


def main():
    """Run comprehensive diagnostic."""
    print("\n🔍 SDD v4.0 Dependency Diagnostic Report")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Working Directory: {Path.cwd()}")

    # Check dependencies
    golden_path_ok, golden_results = check_golden_path()
    base_ok, base_results = check_base_dependencies()

    # Check platform
    platform_info = check_platform()

    # Check environment
    env_info = check_environment()

    # Provide recommendations
    recommendations = provide_recommendations(golden_path_ok, base_ok, platform_info)

    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'working_directory': str(Path.cwd()),
        'golden_path_satisfied': golden_path_ok,
        'base_dependencies_satisfied': base_ok,
        'platform': platform_info,
        'environment': env_info,
        'golden_path_results': golden_results,
        'base_results': base_results,
        'recommendations': recommendations,
        'sdd_compliance': {
            'ready_for_production': golden_path_ok and not os.environ.get('USE_FALLBACKS'),
            'can_use_fallbacks': base_ok,
            'validation_mode': env_info.get('CONFIG_VALIDATION_MODE', 'STRICT (default)'),
        }
    }

    # Save report
    report_path = Path('dependency_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Report saved to {report_path}")

    # Exit with appropriate code
    if golden_path_ok:
        print("\n🎉 System ready for SDD v4.0 Golden Path!")
        sys.exit(0)
    else:
        print(f"\n⚠️  System not ready for Golden Path. See recommendations above.")
        sys.exit(1)


if __name__ == "__main__":
    main()