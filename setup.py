from setuptools import setup

setup(name='processes',
      version='0.1',
      description='Generates realizations of common, multivariate stochastic processes',
      url='https://github.com/dirkbrockmann/processes',
      author='Dirk Brockmann',
      author_email='dirk.brockmann@hu-berlin.de',
      license='MIT',
      packages=['processes'],
      install_requires=[
                'numpy'
            ],
      zip_safe=False)