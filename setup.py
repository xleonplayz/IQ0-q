"""
Setup script for the simos_nv_simulator package.
"""

from setuptools import setup, find_packages

# Load long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='simos_nv_simulator',
    version='0.1.0',
    description='NV-Center simulator with SimOS integration for Qudi',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='IQO Team',
    author_email='noreply@example.com',
    url='https://github.com/xleonplayz/IQ0-q',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',  # Required for quantum state evolution
        # SimOS is optional, will be used if available
        # If not available, a placeholder implementation will be used
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.12.0',
            'flake8>=3.9.0',
            'black>=21.5b2',
            'mypy>=0.812',
            'types-scipy',  # Type stubs for scipy
        ],
    },
)