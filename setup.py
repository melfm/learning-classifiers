from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The playground repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

setup(
    name='playground',
    py_modules=['playground'],
    install_requires=[
        'cloudpickle==0.5.2',
        'gym[atari,box2d,classic_control]>=0.10.8',
        'ipython',
        'joblib',
        'matplotlib',
        'numpy',
        'pandas',
        'pytest',
        'scipy',
        'seaborn==0.8.1',
        'tensorflow>=1.8.0',
        'tqdm'
    ],
    description="RL Playground.",
    author="Melissa Mozifian",
)
