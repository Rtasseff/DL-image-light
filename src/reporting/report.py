"""
Training and evaluation report generation module.

This module provides utilities for generating comprehensive HTML reports
that include training curves, metrics, visualizations, and model information.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import base64
import io

import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template

from ..utils.logging import get_logger


def encode_image_to_base64(image_path: Path) -> str:
    """
    Encode image file to base64 string for HTML embedding.

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded image string
    """
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        encoded = base64.b64encode(image_data).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""


def load_metrics_from_csv(metrics_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load training metrics from CSV files.

    Args:
        metrics_dir: Directory containing metrics CSV files

    Returns:
        DataFrame with metrics or None if not found
    """
    logger = get_logger(__name__)

    # Look for Lightning logs CSV
    csv_files = list(metrics_dir.glob("**/*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {metrics_dir}")
        return None

    # Use the first CSV file found
    csv_path = csv_files[0]
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        logger.error(f"Failed to load metrics CSV {csv_path}: {e}")
        return None


def create_training_summary(config: Dict[str, Any], metrics_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Create training summary information.

    Args:
        config: Training configuration
        metrics_df: Training metrics DataFrame

    Returns:
        Dictionary with summary information
    """
    summary = {
        'project_name': config.get('project_name', 'N/A'),
        'model_architecture': config.get('model', {}).get('architecture', 'N/A'),
        'encoder': config.get('model', {}).get('encoder', 'N/A'),
        'dataset': config.get('dataset', {}).get('name', 'N/A'),
        'epochs': config.get('training', {}).get('epochs', 'N/A'),
        'batch_size': config.get('training', {}).get('batch_size', 'N/A'),
        'learning_rate': config.get('training', {}).get('learning_rate', 'N/A'),
        'loss_function': config.get('training', {}).get('loss', {}).get('type', 'N/A'),
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    if metrics_df is not None and not metrics_df.empty:
        # Get final metrics
        final_metrics = {}
        for col in metrics_df.columns:
            if 'val_' in col or 'train_' in col:
                final_value = metrics_df[col].dropna().iloc[-1] if not metrics_df[col].dropna().empty else None
                if final_value is not None:
                    final_metrics[col] = float(final_value)

        summary['final_metrics'] = final_metrics
        summary['total_epochs_trained'] = len(metrics_df.dropna(subset=['epoch'])) if 'epoch' in metrics_df.columns else 'N/A'
    else:
        summary['final_metrics'] = {}
        summary['total_epochs_trained'] = 'N/A'

    return summary


def generate_training_report(
    run_dir: Path,
    output_path: Optional[Path] = None,
    include_images: bool = True
) -> Path:
    """
    Generate comprehensive training report.

    Args:
        run_dir: Training run directory
        output_path: Output path for report HTML
        include_images: Whether to embed images in report

    Returns:
        Path to generated report
    """
    logger = get_logger(__name__)
    logger.info(f"Generating training report for {run_dir}")

    if output_path is None:
        output_path = run_dir / "training_report.html"

    report_data = {
        'title': 'Training Report',
        'run_directory': str(run_dir),
        'summary': {},
        'metrics': {},
        'images': {},
        'files': {}
    }

    # Load configuration
    config_path = run_dir / "config.yaml"
    config = {}
    if config_path.exists():
        from ..core.config import load_config
        try:
            config = load_config(str(config_path))
            logger.info("Loaded training configuration")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    # Load metrics
    metrics_df = None
    metrics_dir = run_dir / "lightning_logs"
    if metrics_dir.exists():
        metrics_df = load_metrics_from_csv(metrics_dir)
        if metrics_df is not None:
            logger.info("Loaded training metrics")

    # Create summary
    report_data['summary'] = create_training_summary(config, metrics_df)

    # Add metrics data
    if metrics_df is not None:
        # Convert metrics to JSON-serializable format
        metrics_dict = {}
        for col in metrics_df.columns:
            if col in ['epoch', 'step'] or 'val_' in col or 'train_' in col:
                series_data = metrics_df[col].dropna()
                if not series_data.empty:
                    metrics_dict[col] = series_data.tolist()

        report_data['metrics'] = metrics_dict

    # Collect images if requested
    if include_images:
        image_dirs = ['plots', 'overlays', 'visualizations']
        for img_dir in image_dirs:
            img_path = run_dir / img_dir
            if img_path.exists():
                images = []
                for img_file in img_path.glob("*.png"):
                    encoded_img = encode_image_to_base64(img_file)
                    if encoded_img:
                        images.append({
                            'name': img_file.name,
                            'data': encoded_img,
                            'caption': img_file.stem.replace('_', ' ').title()
                        })

                if images:
                    report_data['images'][img_dir] = images
                    logger.info(f"Embedded {len(images)} images from {img_dir}")

    # Collect file information
    important_files = ['checkpoints', 'final_metrics.json', 'dataset_info.json']
    for file_pattern in important_files:
        if file_pattern == 'checkpoints':
            ckpt_dir = run_dir / "checkpoints"
            if ckpt_dir.exists():
                ckpt_files = list(ckpt_dir.glob("*.ckpt"))
                report_data['files']['checkpoints'] = [f.name for f in ckpt_files]
        else:
            file_path = run_dir / file_pattern
            if file_path.exists():
                report_data['files'][file_pattern] = str(file_path)

    # Generate HTML report
    html_content = create_html_report(report_data)

    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"Training report saved to {output_path}")
    return output_path


def generate_evaluation_report(
    eval_dir: Path,
    output_path: Optional[Path] = None,
    include_images: bool = True
) -> Path:
    """
    Generate comprehensive evaluation report.

    Args:
        eval_dir: Evaluation results directory
        output_path: Output path for report HTML
        include_images: Whether to embed images in report

    Returns:
        Path to generated report
    """
    logger = get_logger(__name__)
    logger.info(f"Generating evaluation report for {eval_dir}")

    if output_path is None:
        output_path = eval_dir / "evaluation_report.html"

    report_data = {
        'title': 'Evaluation Report',
        'eval_directory': str(eval_dir),
        'summary': {},
        'metrics': {},
        'images': {},
        'files': {}
    }

    # Load overall metrics
    overall_metrics_path = eval_dir / "overall_metrics.json"
    if overall_metrics_path.exists():
        try:
            with open(overall_metrics_path, 'r') as f:
                overall_metrics = json.load(f)
            report_data['summary'] = overall_metrics
            logger.info("Loaded evaluation metrics")
        except Exception as e:
            logger.warning(f"Failed to load overall metrics: {e}")

    # Load per-image metrics
    per_image_path = eval_dir / "per_image_metrics.csv"
    if per_image_path.exists():
        try:
            per_image_df = pd.read_csv(per_image_path)
            # Create summary statistics
            summary_stats = per_image_df[['dice_score', 'iou', 'precision', 'recall', 'accuracy']].describe()
            report_data['metrics']['per_image_summary'] = summary_stats.to_dict()
            report_data['metrics']['total_images'] = len(per_image_df)
            logger.info("Loaded per-image metrics")
        except Exception as e:
            logger.warning(f"Failed to load per-image metrics: {e}")

    # Collect visualization images
    if include_images:
        vis_dir = eval_dir / "visualizations"
        if vis_dir.exists():
            images = []
            for img_file in vis_dir.glob("*.png"):
                encoded_img = encode_image_to_base64(img_file)
                if encoded_img:
                    images.append({
                        'name': img_file.name,
                        'data': encoded_img,
                        'caption': img_file.stem.replace('_', ' ').title()
                    })

            if images:
                report_data['images']['visualizations'] = images
                logger.info(f"Embedded {len(images)} evaluation visualizations")

        # Include confusion matrix if available
        cm_path = eval_dir / "confusion_matrix.png"
        if cm_path.exists():
            encoded_cm = encode_image_to_base64(cm_path)
            if encoded_cm:
                report_data['images']['confusion_matrix'] = [{
                    'name': 'confusion_matrix.png',
                    'data': encoded_cm,
                    'caption': 'Confusion Matrix'
                }]

    # Add file information
    report_data['files'] = {
        'overall_metrics': str(overall_metrics_path) if overall_metrics_path.exists() else None,
        'per_image_metrics': str(per_image_path) if per_image_path.exists() else None,
        'metrics_summary': str(eval_dir / "metrics_summary.csv") if (eval_dir / "metrics_summary.csv").exists() else None
    }

    # Generate HTML report
    html_content = create_html_report(report_data, report_type='evaluation')

    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"Evaluation report saved to {output_path}")
    return output_path


def create_html_report(report_data: Dict[str, Any], report_type: str = 'training') -> str:
    """
    Create HTML report from report data.

    Args:
        report_data: Dictionary containing report data
        report_type: Type of report ('training' or 'evaluation')

    Returns:
        HTML content as string
    """
    # This is a basic HTML template - in a real implementation,
    # you would use a proper template file
    template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        .header {
            background-color: #f4f4f4;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .section {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        .metrics-table th, .metrics-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .metrics-table th {
            background-color: #f2f2f2;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .image-item {
            text-align: center;
        }
        .image-item img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .metric-value {
            font-weight: bold;
            color: #2c5aa0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p><strong>Generated:</strong> {{ summary.generation_time if summary.generation_time else 'N/A' }}</p>
        {% if report_type == 'training' %}
        <p><strong>Run Directory:</strong> {{ run_directory }}</p>
        {% else %}
        <p><strong>Evaluation Directory:</strong> {{ eval_directory }}</p>
        {% endif %}
    </div>

    {% if summary %}
    <div class="section">
        <h2>Summary</h2>
        <table class="metrics-table">
            {% if report_type == 'training' %}
            <tr><td>Project Name</td><td>{{ summary.project_name }}</td></tr>
            <tr><td>Model Architecture</td><td>{{ summary.model_architecture }}</td></tr>
            <tr><td>Encoder</td><td>{{ summary.encoder }}</td></tr>
            <tr><td>Dataset</td><td>{{ summary.dataset }}</td></tr>
            <tr><td>Epochs</td><td>{{ summary.epochs }}</td></tr>
            <tr><td>Batch Size</td><td>{{ summary.batch_size }}</td></tr>
            <tr><td>Learning Rate</td><td>{{ summary.learning_rate }}</td></tr>
            <tr><td>Loss Function</td><td>{{ summary.loss_function }}</td></tr>
            {% if summary.total_epochs_trained %}
            <tr><td>Total Epochs Trained</td><td>{{ summary.total_epochs_trained }}</td></tr>
            {% endif %}
            {% else %}
            {% if summary.threshold %}
            <tr><td>Threshold</td><td>{{ summary.threshold }}</td></tr>
            {% endif %}
            {% if summary.summary %}
            {% for key, value in summary.summary.items() %}
            <tr><td>{{ key.replace('_', ' ').title() }}</td><td class="metric-value">{{ "%.4f"|format(value) if value is number else value }}</td></tr>
            {% endfor %}
            {% endif %}
            {% endif %}
        </table>

        {% if summary.final_metrics and report_type == 'training' %}
        <h3>Final Training Metrics</h3>
        <table class="metrics-table">
            {% for key, value in summary.final_metrics.items() %}
            <tr><td>{{ key.replace('_', ' ').title() }}</td><td class="metric-value">{{ "%.4f"|format(value) }}</td></tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>
    {% endif %}

    {% if metrics %}
    <div class="section">
        <h2>Metrics</h2>
        {% if metrics.per_image_summary %}
        <h3>Per-Image Statistics</h3>
        <p><strong>Total Images:</strong> {{ metrics.total_images }}</p>
        <table class="metrics-table">
            <tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>25%</th><th>50%</th><th>75%</th><th>Max</th></tr>
            {% for metric, stats in metrics.per_image_summary.items() %}
            <tr>
                <td>{{ metric.replace('_', ' ').title() }}</td>
                <td>{{ "%.4f"|format(stats.mean) }}</td>
                <td>{{ "%.4f"|format(stats.std) }}</td>
                <td>{{ "%.4f"|format(stats.min) }}</td>
                <td>{{ "%.4f"|format(stats['25%']) }}</td>
                <td>{{ "%.4f"|format(stats['50%']) }}</td>
                <td>{{ "%.4f"|format(stats['75%']) }}</td>
                <td>{{ "%.4f"|format(stats.max) }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>
    {% endif %}

    {% if images %}
    {% for section_name, image_list in images.items() %}
    <div class="section">
        <h2>{{ section_name.replace('_', ' ').title() }}</h2>
        <div class="image-grid">
            {% for image in image_list %}
            <div class="image-item">
                <img src="{{ image.data }}" alt="{{ image.name }}">
                <p><strong>{{ image.caption }}</strong></p>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endfor %}
    {% endif %}

    {% if files %}
    <div class="section">
        <h2>Available Files</h2>
        <ul>
            {% for file_type, file_path in files.items() %}
            {% if file_path %}
            <li><strong>{{ file_type.replace('_', ' ').title() }}:</strong> {{ file_path }}</li>
            {% endif %}
            {% endfor %}
        </ul>
    </div>
    {% endif %}

    <div class="section">
        <p><em>Report generated by DL-Image-Light Segmentation Platform</em></p>
    </div>
</body>
</html>
    """

    template = Template(template_str)
    return template.render(**report_data, report_type=report_type)