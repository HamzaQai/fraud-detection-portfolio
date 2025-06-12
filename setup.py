"""
Setup configuration for fraud detection portfolio
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fraud-detection-portfolio",
    version="1.0.0",
    author="Votre Nom",
    author_email="votre.email@domain.com",
    description="Portfolio professionnel de détection de fraude avec solutions Kaggle",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/votre-username/fraud-detection-portfolio",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "isort>=5.0",
        ],
        "viz": [
            "plotly>=5.0",
            "dash>=2.0",
            "streamlit>=1.0",
        ],
        "deep": [
            "tensorflow>=2.8",
            "torch>=1.11",
            "transformers>=4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "fraud-train=src.models.train:main",
            "fraud-predict=src.models.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["configs/*.yaml", "configs/*.json"],
    },
)
