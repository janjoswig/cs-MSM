from setuptools import Extension, find_packages, setup

from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        "csmsm.estimator", ["src/csmsm/estimator.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c++",
        include_dirs=[np.get_include()]
        )
]

compiler_directives = {
    "language_level": 3,
    "embedsignature": True,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "nonecheck": False
    }

extensions = cythonize(extensions, compiler_directives=compiler_directives)

with open("README.md", "r") as readme:
    desc = readme.read()

sdesc = "A Python package for core-set Markov-state model estimation"

requirements_map = {
    "mandatory": "",
    "test": "-test"
    }

requirements = {}
for category, fname in requirements_map.items():
    with open(f"requirements{fname}.txt") as fp:
        requirements[category] = fp.read().strip().split("\n")

setup(
    name='csmsm',
    version='0.0.1',
    keywords=["core-set", "Markov-model"],
    author="Jan-Oliver Joswig",
    author_email="jan.joswig@fu-berlin.de",
    description=sdesc,
    long_description=desc,
    long_description_content_type="text/markdown",
    url="https://github.com/janjoswig/cs-MSM",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    ext_modules=extensions,
    install_requires=requirements["mandatory"],
    extras_require={
        "test": requirements["test"],
        },
    zip_safe=False
    )
