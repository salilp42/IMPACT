from setuptools import setup, find_packages

setup(
    name="impact-fmri",
    version="0.2.0",
    description="IMPACT: Integrative Multimodal Pipeline for Advanced Connectivity and Time-series",
    author="Salil Patel",
    author_email="salilp42@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "nibabel>=3.2.0",
        "nilearn>=0.8.0",
        "scikit-learn>=0.24.0",
        "torch>=1.10.0",
        "tqdm>=4.50.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
) 