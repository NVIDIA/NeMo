import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nemo_cv",
    version="0.1",
    author="NVIDIA",
    author_email="nemo-toolkit@nvidia.com",
    description="Collection of Neural Modules for Computer Vision",
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
        'nemo_toolkit',
        'torch==1.2.0',
        'torchvision==0.4.0'
    ]
)
