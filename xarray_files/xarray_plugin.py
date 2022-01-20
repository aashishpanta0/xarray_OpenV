import os.path
import numpy as np
import xarray as xr
from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
    BackendArray,
    BackendEntrypoint,
)
from xarray.core import indexing
from xarray.core.indexing import ExplicitIndexer, LazilyIndexedArray


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


class OpenVisusEntryPoint(BackendEntrypoint):
 

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
        backend_array=OpenVisusBackendArray()
        """
        
        """
        data = LazilyIndexedArray(backend_array)
        """ can select the variables and dimensions by updating the xr.variable(data)
        to xr.variable(dims,attrs,data)
        """
        vars= xr.Variable( data)
        vars.set_close(self.close_method)
        return vars
    
    def close_method(self):
        return
    
      

    def guess_can_open(self, filename_or_obj):  # pylint: disable=arguments-renamed
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext[1:].lower() in CAN_OPEN_EXTS

# subclass implementation of BackendArray
class OpenVisusBackendArray(BackendArray):
    def __init__(
        self,
        shape,
        dtype,
        lock,
        # other backend specific keyword arguments
    ):
        self.shape = shape
        self.dtype = lock
        self.lock = dtype

    def __getitem__(
        self, key: ExplicitIndexer
    ) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

