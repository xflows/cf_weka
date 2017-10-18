from setuptools import setup, find_packages
import os

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
]

dist = setup(
    name='cf_weka',
    version='0.1',
    author='Darko Aleksovski',
    description='Package providing data mining widgets for ClowdFlows 2.0, based on WEKA.',
    url='https://github.com/xflows/cf_weka',
    license='MIT License',
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'cf_core',
        'cf_data_mining',
        'JPype1',
        'scipy',
        'numpy',
        'scikit-learn==0.15.2'
    ]
)
