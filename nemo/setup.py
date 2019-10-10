import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nemo_toolkit",
    version="0.8",
    author="NVIDIA",
    author_email="nemo-toolkit@nvidia.com",
    description="NEMO core package. Necessary for all collections.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nvidia/nemo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License"
    ],
    install_requires=[
        'torch==1.2.0',
        'torchvision',
        'tensorboardX',
        'pandas',
        'wget'
    ]
)

