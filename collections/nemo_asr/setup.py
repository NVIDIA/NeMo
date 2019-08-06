import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nemo_asr",
    version="0.0.1",
    author="AI Applications @ NVIDIA",
    author_email="okuchaiev@nvidia.com",
    description="Collection of Neural Modules for Speech Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nvidia/nemo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'nemo',
        'toml',
        'librosa',
        'num2words',
        'inflect',
        'torch-stft',
        'soundfile',
        'marshmallow',
        'ruamel.yaml'
    ]
)
