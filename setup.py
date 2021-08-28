from setuptools import setup, Extension
import setuptools
import os
import sys

# get __version__, __author__, and __email__
exec(open("./processes/metadata.py").read())

# I would rename the package, because "processes" is already taken by some other package on PyPI.
# Which means you cannot publish it on PyPI in this state. Which means
#
#    pip install processes
#
# won't work.
setup(
    name='processes',
    version=__version__,
    author=__author__,
    author_email=__email__,
    url='https://github.com/dirkbrockmann/processes',
    license=__license__,
    description="Generates realizations of common, multivariate stochastic processes",
    long_description='',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
                       'numpy>=1.17',
                ],
    tests_require=['pytest', 'pytest-cov'],
    setup_requires=['pytest-runner'],
    classifiers=['License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 ],
    project_urls={
        'Documentation': 'http://processes.readthedocs.io',
        'Contributing Statement': 'https://github.com/dirkbrockmann/processes/blob/master/CONTRIBUTING.md',
        'Bug Reports': 'https://github.com/dirkbrockmann/processes/issues',
        'Source': 'https://github.com/dirkbrockmann/processes/',
        'PyPI': 'https://pypi.org/project/komoog/',
    },
    include_package_data=True,
    zip_safe=False,
)
