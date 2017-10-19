from setuptools import setup

setup(name='superconductivity',
      version='0.1',
      description='Useful tools for superconductivity',
      author='Mikel Rouco',
      author_email='mrouco001@ikasle.ehu.eus',
      license='MIT',
      packages=['superconductivity'],
      install_requires = [
          'numpy',
          'scipy'
          ],
      zip_safe=False)
