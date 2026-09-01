from setuptools import setup, find_packages

setup(
    name="viveka",
    version="1.0.0",
    description="Cross-Paper Hypothesis Generation via Experiment-Similarity Graphs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Divyansh Bhatia, Dhiyanesh Sidhaiyan",
    url="https://github.com/dnbresearch/VIVEKA",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "anthropic>=0.30.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "networkx>=3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "requests>=2.28.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "validation": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.30.0",
            "datasets>=2.14.0",
            "peft>=0.5.0",
        ],
        "graph": [
            "torch-geometric>=2.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "viveka-fetch=viveka.fetch_papers:main",
            "viveka-mine=viveka.scale_evaluation:main",
            "viveka-community=viveka.community_detection:main",
            "viveka-generate=viveka.hypothesis_generation:main",
            "viveka-evaluate=viveka.evaluate:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
