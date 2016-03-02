from setuptools import setup

setup(name='molpy',
      version='1.0',
      description='A package to handle Molcas wavefunction data',
      license='GNU GPLv2',
      author='Steven Vancoillie',
      author_email='molpy@steven.se',
      url='https://github.com/steabert/molpy/',
      packages=['molpy'],
      scripts=['penny'],
      requires=['numpy'],
     )
