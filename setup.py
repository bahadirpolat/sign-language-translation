from setuptools import setup

setup(
    name='signpiece',
    version='0.0.1',    
    description='A package for sign language translation',
    url='https://github.com/karahan-sahin/signpiece',
    author='Karahan Sahin',
    author_email='karahan.sahin@boun.edu.tr',
    license='BSD 2-clause',
    packages=['signpiece'],
    install_requires=[
        # 'mpi4py>=2.0',
        'numpy',  
        'torch',
        'transformers',
        'sentencepiece',
        'evaluate',
        'h5py',
        'pypmi',
        # 'bluert'
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.7s',
    ],
)