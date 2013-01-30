# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as stream:
    long_desc = stream.read()

requires = ['Sphinx>=0.6', 'pyparsing']

setup(
    name='sphinxcontrib-doxylink',
    version='1.3',
	url='http://packages.python.org/sphinxcontrib-doxylink',
    download_url='http://pypi.python.org/pypi/sphinxcontrib-doxylink',
    license='BSD',
    author='Matt Williams',
    author_email='matt@milliams.com',
    description='Sphinx extension for linking to Doxygen documentation.',
    long_description=long_desc,
    keywords=['sphinx','doxygen','documentation','c++'],
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Documentation',
        'Topic :: Utilities',
    ],
    platforms='any',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requires,
    namespace_packages=['sphinxcontrib'],
	test_suite="tests",
	use_2to3=True,
)
