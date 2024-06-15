from setuptools import setup, find_packages
# Setup.py file for the package


setup(
    name='thetesterpackage',
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
    # author_email='your_email@example.com',
    description='A basic package',
    # url='https://github.com/your_username/my_package',
)
