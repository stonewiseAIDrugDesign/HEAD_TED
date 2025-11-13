from setuptools import setup, find_packages

setup(
    name='hea-detect',                    
    version='0.1.0',
    description='High-Energy Atom Detection toolkit',
    author='Bin XI',
    author_email='binxi_@outlook.cn',
    packages=find_packages(),              
    package_data={
        'hea_detector': ['torchani/**/*'], 
    },
    include_package_data=True,
    install_requires=[
        'torch',
        'rdkit==2023.9.4',
        'numpy<2',
        'matplotlib',
        'pandas',
        'scipy',
        'lark==1.1.9',
        'importlib_metadata',
        'requests'
    ],
    entry_points={
        'console_scripts': [
            'hea-detect = hea_detector.main:main'
        ]
    },
)