import setuptools 
from setuptools import setup 

setup(
    name='nnf2',
    version='v0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/mgjeon/project',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Mingyu Jeon',
    author_email='',
    description='',
    install_requires=['torch>=1.8', 'sunpy[all]>=3.0', 'scikit-image', 'scikit-learn', 'tqdm',
                      'numpy', 'matplotlib', 'astropy', 'drms', 'wandb>=0.13', 'lightning==1.9.3', 'pytorch_lightning==1.9.3'],
)