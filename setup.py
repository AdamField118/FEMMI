"""
Setup script for weak_lensing_poisson package

Install in development mode:
    pip install -e .

Or regular install:
    pip install .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text()
else:
    long_description = "JAX-based FEM for weak gravitational lensing"

setup(
    name="weak_lensing_poisson",
    version="0.1.0",
    author="Adam Field",
    author_email="adfield@wpi.edu",
    description="GPU-accelerated FEM solver for weak gravitational lensing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AdamField118/computational_physics_isp",
    
    # Package discovery
    packages=find_packages(),
    package_dir={"": "."},
    
    # Dependencies
    python_requires=">=3.10",
    install_requires=[
        "jax>=0.4.20",
        "jaxlib>=0.4.20",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
    ],
    
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
        ],
        "mesh": [
            "triangle>=20200424",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)