import os

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

requirements = [
    'torch',
    'omegaconf',
    'torchvision',
    'tqdm',
    'numpy',
    'matplotlib',
    'timm',
    'webdataset',
    'wandb',
    'opencv-python',
    'transformers',
    'hydra-core',
]

setup(
    name='featsharp',
    version='0.0.1',
    packages=find_packages(),
    install_requires=requirements,
    author='Mike Ranzinger',
    author_email='mranzinger@nvidia.com',
    description='FeatSharp: Your Vision Model Features, Sharper',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/NVlabs/FeatSharp',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: NVSC License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    cmdclass={
        'build_ext': BuildExtension
    }
)
