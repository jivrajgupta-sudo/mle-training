from setuptools import find_packages, setup

setup(
    name="housing-project",
    version="0.1.0",
    description="A packageable California housing price workflow",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "scikit-learn>=1.2",
    ],
    entry_points={
        "console_scripts": [
            "housing-ingest=housing_project.cli:ingest_main",
            "housing-train=housing_project.cli:train_main",
            "housing-score=housing_project.cli:score_main",
        ]
    },
)
