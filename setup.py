from setuptools import setup, find_packages

setup(
    name="trawl-uq",
    version="0.1.0",
    description="Uncertainty-Guided Tensor Decomposition for LLMs",
    author="Het Patel",
    author_email="hpate061@ucr.edu",
    url="https://github.com/HettyPatel/trawl-uq",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "tensorly>=0.8.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "datasets>=2.14.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
)
