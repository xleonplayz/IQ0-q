from setuptools import setup, find_packages

setup(
    name="nv_simulator",
    version="0.1.0",
    description="NV Center Quantum Simulator for Qudi",
    author="IQO Team",
    author_email="user@example.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "qutip>=4.6.0",
    ],
    python_requires=">=3.9, <3.11",
)