#!/usr/bin/env python3
"""
Setup script for ML-Based Test Automation
"""

from setuptools import setup, find_packages
import os

# Get long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from version.py
about = {}
with open(os.path.join("src", "version.py"), "r", encoding="utf-8") as f:
    exec(f.read(), about)

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f.readlines() if not line.startswith("#")]

setup(
    name="ml-test-automation",
    version=about["__version__"],
    author="ML Test Automation Team",
    author_email="your.email@example.com",
    description="Machine Learning based Test Automation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/ML-Based-Test-Automation",
    project_urls={
        "Bug Tracker": "https://github.com/YOUR_USERNAME/ML-Based-Test-Automation/issues",
        "Documentation": "https://github.com/YOUR_USERNAME/ML-Based-Test-Automation/docs",
        "Source Code": "https://github.com/YOUR_USERNAME/ML-Based-Test-Automation",
    },
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ml-test-automation=main:main",
        ],
    },
) 