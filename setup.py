from setuptools import setup

setup(name='currpy',
      version='0.2',
      description='Useful tools for dirty superconductivity',
      author='Mikel Rouco',
      author_email='mrouco001@ikasle.ehu.eus',
      license='MIT',
      packages=['currpy'],
      install_requires = [
          'numpy',
          'scipy'
          ],
      zip_safe=False)
