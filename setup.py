# setup.py is responsible for creating the machine learning application as a package that can be easily installed and distributed.@
# It defines the package metadata, dependencies, and other configurations needed for packaging and distribution.

from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path)-> List[str]:
    '''
    This function reads the requirements from the specified file and returns a list of dependencies.
    '''
    HYPHEN_E_DOT = '-e .'
    requirements =[]
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements =[req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='mlproject',
    version='0.1.0',
    author='Sankha',
    author_email='sankha091@gmail.com',
    description='First project of ML Ops',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)