import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nemo_nlp",
    version="0.3",
    author="NVIDIA",
    author_email="okuchaiev@nvidia.com",
    description="Collection of Neural Modules for Natural Language Processing",
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
        'nemo_toolkit',
        'torchtext',
        'sentencepiece',
        'boto3',
        'unidecode',
        'pytorch-transformers',
        'matplotlib',
        'youtokentome'
    ]
)
