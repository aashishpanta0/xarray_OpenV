import os.path
import numpy as np
import xarray as xr
from xarray.backends import BackendEntrypoint
from xarray.backends import BackendArray
from xarray.core import indexing



# all extensions that can be loaded (must be changed later)
CAN_OPEN_EXTS = {
    "asc",
    "geotif",
    "geotiff",
    "img",
    "j2k",
    "jp2",
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "vrt",
}


class NewBackendPoint(BackendEntrypoint):
 

    def open_dataset(
        self,
        filename_or_obj,
        drop_variables=None,
        parse_coordinates=None,
        chunks=None,
        cache=None,
        lock=None,
        masked=False,
        mask_and_scale=True,
        variable=None,
        group=None,
        default_name="band_data",
        decode_times=True,
        decode_timedelta=None,
        open_kwargs=None,
    ):
        open_dataset_parameters = ["filename_or_obj", "drop_variables"]
      

    def guess_can_open(self, filename_or_obj):  # pylint: disable=arguments-renamed
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext[1:].lower() in CAN_OPEN_EXTS



#indexing here

