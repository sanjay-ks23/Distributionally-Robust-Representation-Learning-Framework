"""
DRRL Framework Setup Configuration.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="drrl-framework",
    version="1.0.0",
    author="DRRL Team",
    author_email="drrl@example.com",
    description="Distributionally Robust Representation Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/drrl-framework",
    packages=find_packages(exclude=["tests", "notebooks", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "drrl-train=scripts.train:main",
            "drrl-eval=scripts.evaluate:main",
        ],
    },
)
