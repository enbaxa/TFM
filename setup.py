from setuptools import setup, find_packages
# Setup.py file for the package


setup(
    name='nnBuilder',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        "torch",
        "pandas",
        "transformers",
        "matplotlib",
        "seaborn"
    ],
    author='Enric Basso',
    description='A basic package',
)
