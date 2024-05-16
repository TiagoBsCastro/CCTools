from setuptools import setup, find_packages

setup(
    name='CCTools',
    version='0.1',
    description='A collection of tools for cosmological computations',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/CCTools',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pyDOE',
        'camb',
        'baccoemu',
        'tensorflow',
        'mpi4py',
        'matplotlib',
        'torch',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)


