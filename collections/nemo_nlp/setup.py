import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nemo_nlp",
    version="0.9.0",
    author="NVIDIA",
    author_email="nemo-toolkit@nvidia.com",
    description="Collection of Neural Modules for Natural Language Processing",
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
        'torchtext',
        'sentencepiece',
        'python-dateutil<2.8.1,>=2.1',
        'boto3',
        'unidecode',
        'transformers',
        'matplotlib',
        'h5py',
        'youtokentome'
    ]
)
