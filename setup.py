from setuptools import find_packages, setup

setup(
    entry_points={
        # set this to the location of core xarray backend
<<<<<<< HEAD
        "xarray.backends": ["openvis_engine=xarray_files.backend:OpenVisusEntryPoint"]
=======
        "xarray.backends": ["openvis=xarray_files.xarray_plugin:NewBackendPoint"]
>>>>>>> 7be4d8f76d7547cbe37a11dec64cf76f2fe4fa5c
        }

)
