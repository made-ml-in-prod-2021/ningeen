from setuptools import find_packages, setup

setup(
    name='ml_project',
    packages=find_packages(),
    version='0.0.1',
    description='First HW ML in prod',
    author='Ruslan Sakaev',
    install_requires=[
        "pytest == 0.0.0",
        "marshmallow_dataclass == 8.4.1",
        "Faker == 8.1.2",
        "pandas == 1.1.3",
        "click == 7.1.2",
        "numpy == 1.19.2",
        "py == 1.9.0",
        "dataclasses == 0.6",
        "PyYAML == 5.3.1",
        "scikit_learn == 0.23.2",
    ],
    license='MIT',
)