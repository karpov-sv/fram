#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

from setuptools import setup, find_packages

requirements = [
    'numpy',
    'scipy',
    'astropy',
    'matplotlib',
    'sep',
    'statsmodels',
    'esutil',
    'psycopg2-binary',
    'ephem',
    'supersmoother',
    # For web interface
    'scikit-build', # needed for line-profiler below?..
    'django',
    # 'django-debug-toolbar',
    # 'django-debug-toolbar-line-profiler',
    'django-el-pagination',
    'markdown',
]

setup(
    name='fram',
    version='0.1',
    description='FRAM telescope related codes',
    author='Sergey Karpov',
    author_email='karpov.sv@gmail.com',
    url='',

    install_requires=requirements,
    packages=['fram'],
    package_dir={'':'src'}, # src/fram, symlink to fram, to behave nicely with development install
)
