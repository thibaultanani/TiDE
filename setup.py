from setuptools import setup, find_packages

setup(
    name='ml-feature-selection',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'mrmr-selection',
        'numpy',
        'openpyxl',
        'pandas',
        'psutil',
        'scikit-learn',
        'scipy',
        'skrebate',
    ],
    author='Thibault Anani',
    author_email='thuny.ta@gmail.com',
    description='Implements multiple type of filter methods and heuristics for the feature selection problem in '
                'machine learning as well as a new one: tournament in differential evolution',
    url='https://github.com/thibaultanani/Tournament-in-Differential-Evolution',
)