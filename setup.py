from setuptools import setup, find_packages

setup(
    name="segmentation-platform",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "segmentation-models-pytorch>=0.3.3",
        "torchmetrics>=0.11.0",
        "albumentations>=1.3.0",
        "timm>=0.9.0",
        "pydantic>=2.0.0",
        "PyYAML>=6.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "opencv-python>=4.8.0",
    ],
    python_requires=">=3.8",
)