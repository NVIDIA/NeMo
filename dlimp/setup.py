from setuptools import find_packages, setup

setup(
    name="dlimp",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow==2.15.0",
        "tensorflow_datasets>=4.9.2",
    ],
    extras_require={
        "convert": [
            "tqdm",
            "tqdm-multiprocess==0.0.11",
        ],
        "dev": [
            "pre-commit==3.3.3",
        ],
    },
)
