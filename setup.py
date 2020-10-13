from setuptools import setup, find_packages

with open("requirements.txt", "r") as r:
    req = r.readlines()

setup(
    name="synthetic_table_metrics",
    version="0.1dev",
    packages=find_packages(),
    install_requires=req,
)
