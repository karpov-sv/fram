#!/usr/bin/env python3

from setuptools import setup

requirements = [
    'setuptools',
    'numpy',
    'scipy',
    'astropy',
    'matplotlib',
    'sep',
    'stdpipe',
    'psycopg2',
    # 'ephem',
    # 'supersmoother',
]

setup(
    name='fram',
    version='0.1',
    description='FRAM telescope related codes',
    author='Sergey Karpov',
    author_email='karpov.sv@gmail.com',
    url='https://github.com/karpov-sv/fram',

    install_requires=requirements,
    packages=['fram'],
)
