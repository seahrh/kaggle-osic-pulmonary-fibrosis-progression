from setuptools import setup, find_packages

__version__ = "1.0"
setup(
    name="kaggle-osic-pfp",
    version=__version__,
    python_requires="~=3.7",
    install_requires=[
        "gcsfs~=0.6.1",
        "google-cloud-logging~=1.15.0",
        "google-cloud-storage~=1.27.0",
        "lightgbm~=2.3.1",
        "pandas~=1.1.1",
        "pyarrow~=0.16.0",
        "pydicom~=2.0.0",
        "scikit-learn~=0.23.2",
        "tensorflow~=2.3.0",
    ],
    extras_require={
        "tests": [
            "black~=19.10b0",
            "mypy>=0.780",
            "pytest>=5.4.2",
            "pytest-cov>=2.9.0",
        ],
        "notebook": [
            "jupyterlab~=1.2.10",
            "seaborn~=0.10.0",
            "tqdm~=4.45.0",
            "imageio~=2.9.0",
        ],
    },
    packages=find_packages("src", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={"": "src"},
    include_package_data=True,
    description="Kaggle OSIC Pulmonary Fibrosis Progression competition 2020",
    license="MIT",
    author="seahrh",
    author_email="seahrh@gmail.com",
    url="https://github.com/seahrh/kaggle-osic-pulmonary-fibrosis-progression",
)
