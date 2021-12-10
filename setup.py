from setuptools import find_packages, setup

setup(
    entry_points={
        "xarray.backends": ["openvis=xarray_files.backend:NewBackendPoint"]
        }

)