from setuptools import find_packages,setup
from typing import List

HYPEN='-e .'
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as  file_obj:
        requirements=file_obj.readlines()
        requirements=[string.replace("\n","") for string in requirements]

        if HYPEN in requirements:
            requirements.remove(HYPEN)
    return requirements
setup(
    name="MLProject",
    version='0.0.1',
    author="Sriram",
    author_email="sriramkoyi15@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)