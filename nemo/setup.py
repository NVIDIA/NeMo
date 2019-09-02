import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nemo_toolkit",
    version="0.3",
    author="NVIDIA",
    author_email="okuchaiev@nvidia.com",
    description="NEMO core package. Necessary for all collections.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nvidia/nemo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache License 2.0"
    ],
    install_requires=[
        'torch>=1.0.0',
        'torchvision',
        'tensorboardX',
        'pandas',
        'wget'
    ]
)

