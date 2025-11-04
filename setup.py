from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Fake-Customer-classifer",
    version="0.1",
    author="Nit Patel",
    packages=find_packages(),
    install_requires = requirements,
)