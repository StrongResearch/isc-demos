from setuptools import setup, find_packages

setup(
    name='cycling_utils',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    extras_require={
        'lightning': ['pytorch-lightning'],
    },
)
