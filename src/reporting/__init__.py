"""
Reporting module for segmentation platform.

This module provides utilities for generating comprehensive
training and evaluation reports.
"""

from .report import (
    generate_training_report,
    generate_evaluation_report,
    create_html_report
)

__all__ = [
    "generate_training_report",
    "generate_evaluation_report",
    "create_html_report"
]