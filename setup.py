from setuptools import find_packages, setup

with open("./requirements.txt", "r") as f:
    requirements = [l.strip() for l in f.readlines() if len(l.strip()) > 0]

setup(
    name="open-eqa-utils",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    author="Enrique Solarte",
    author_email="enrique.solarte.pardo@gmail.com",
    description=("This PKG aims to complement and handle OpenEQA dataset"),
    license="BSD",
)
