from distutils.core import setup


def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()


setup(
    name='titanic',
    version='0.1',
    description='Analysis of the Titanic dataset',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/a2ed/titanic_datascience',
    author='Aiden Arnold',
    author_email='aiden.egf.arnold@gmail.com',
    license='MIT',
    packages=['titanic'],
    install_requires=[
        'pypandoc>=1.4',
        'watermark>=1.5.0',
        'pandas>=0.20.3',
        'scikit-learn>=0.19.0',
        'scipy>=0.19.1',
        'matplotlib>=2.1.0',
        'pytest>=3.2.3',
        'pytest-runner>=2.12.1'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
