from setuptools import find_packages, setup

setup(
    entry_points={
        # set this to the location of core xarray backend
        "xarray.backends": ["openvis=xarray_files.backend:NewBackendPoint"]
        }

)