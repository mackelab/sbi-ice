from setuptools import find_packages, setup

# Package meta-data.
NAME = "sbi_ice"
URL = "https://github.com/mackelab/sbi_ice"
EMAIL = "guy.moss@student.uni-tuebingen.de"
AUTHOR = "Guy Moss"
REQUIRES_PYTHON = ">=3.9.0"

REQUIRED = ["scipy", "numpy", "matplotlib","jupyter","pandas","hydra-core","hydra-submitit-launcher","hydra-joblib-launcher","sbi","tueplots","scikit-learn","pathlib"]

setup(
    name=NAME,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
)