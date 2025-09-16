#!/usr/bin/env python3
"""
Setup script for DeepEval - Enterprise-grade AI Evaluation Framework
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="deepeval",
    version="1.0.0",
    author="DeepEval Team",
    author_email="team@deepeval.ai",
    description="Enterprise-grade AI Evaluation Framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/deepeval/deepeval",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0,<3.0.0",
        "numpy>=1.21.0,<2.0.0",
        "pydantic>=1.10.0,<3.0.0",
        "sqlalchemy>=1.4.0,<3.0.0",
        "fastapi>=0.100.0,<1.0.0",
        "uvicorn[standard]>=0.20.0,<1.0.0",
    ],
    extras_require={
        "llm": [
            "openai>=1.0.0,<2.0.0",
            "anthropic>=0.7.0,<1.0.0",
            "google-generativeai>=0.3.0,<1.0.0",
        ],
        "monitoring": [
            "psutil>=5.9.0,<6.0.0",
        ],
        "dev": [
            "pytest>=7.0.0,<8.0.0",
            "pytest-asyncio>=0.21.0,<1.0.0",
            "pytest-cov>=4.0.0,<5.0.0",
            "black>=22.0.0,<24.0.0",
            "isort>=5.10.0,<6.0.0",
            "flake8>=5.0.0,<7.0.0",
            "mypy>=1.0.0,<2.0.0",
            "pytest-mock>=3.10.0,<4.0.0",
            "pytest-xdist>=3.0.0,<4.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0,<8.0.0",
            "sphinx-rtd-theme>=1.2.0,<3.0.0",
            "myst-parser>=0.18.0,<3.0.0",
        ],
        "production": [
            "openai>=1.0.0,<2.0.0",
            "anthropic>=0.7.0,<1.0.0",
            "google-generativeai>=0.3.0,<1.0.0",
            "psutil>=5.9.0,<6.0.0",
        ],
        "all": [
            "openai>=1.0.0,<2.0.0",
            "anthropic>=0.7.0,<1.0.0",
            "google-generativeai>=0.3.0,<1.0.0",
            "psutil>=5.9.0,<6.0.0",
            "pytest>=7.0.0,<8.0.0",
            "pytest-asyncio>=0.21.0,<1.0.0",
            "pytest-cov>=4.0.0,<5.0.0",
            "black>=22.0.0,<24.0.0",
            "isort>=5.10.0,<6.0.0",
            "flake8>=5.0.0,<7.0.0",
            "mypy>=1.0.0,<2.0.0",
            "pytest-mock>=3.10.0,<4.0.0",
            "pytest-xdist>=3.0.0,<4.0.0",
            "sphinx>=5.0.0,<8.0.0",
            "sphinx-rtd-theme>=1.2.0,<3.0.0",
            "myst-parser>=0.18.0,<3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deepeval=deepeval.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
