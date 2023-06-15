from setuptools import setup

setup(
    name='project_pat',
    url='https://github.com/jladan/package_demo',
    author='https://github.com/joshuanario/project-pat',
    author_email='me@joshuanario.com',
    packages=['project_pat'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'piecewise_regression'],
    version='0.1',
    license='MIT',
    description='python library for some functions detailed in https://www.joshuanario.com/perfeng.html',
    # long_description=open('README.txt').read(),
)