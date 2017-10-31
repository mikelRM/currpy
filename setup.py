from setuptools import setup

setup(name='currpy',
      version='0.1',
      description='Useful tools for superconductivity',
      author='Mikel Rouco',
      author_email='mrouco001@ikasle.ehu.eus',
      license='MIT',
      packages=['currpy'],
      install_requires = [
          'numpy',
          'scipy'
          ],
      zip_safe=False)
