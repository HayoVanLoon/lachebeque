
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow-hub'
]

setup(
    name='lachebeque',
    version='0.1',
    author='Hayo van Loon',
    author_email='hayo.vanloon@incentro.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='A joke rating model',
)
