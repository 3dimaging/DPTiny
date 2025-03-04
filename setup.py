from setuptools import setup, find_packages

setup(
    name='dptiny',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',  # for mnist dataset
    ],
    author='Weizhe',
    description='A minimal implementation of deep learning framework',
)
