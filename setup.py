from setuptools import setup, find_packages

setup(
    name='tide-feature-selection',
    version='1.1.2',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'openpyxl',
        'pandas',
        'psutil',
        'scikit-learn',
        'scipy',
    ],
    author="Thibault Anani, François Delbot, Jean-François Pradat-Peyre",
    author_email="thibault.anani@gmail.com, francois.delbot@lip6.fr, jean-francois.pradat-peyre@lip6.fr",
    maintainer="Thibault Anani",
    maintainer_email="thibault.anani@gmail.com",
    description='Implements multiple type of filter methods and heuristics for the feature selection problem in '
                'machine learning as well as a new one: tournament in differential evolution',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url='https://github.com/thibaultanani/TiDE',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)