import os.path
import numpy as np
import xarray as xr
from OpenVisus import *
from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
    BackendArray,
    BackendEntrypoint,
)
from xarray.core import indexing
from xarray.core.indexing import LazilyIndexedArray
import matplotlib.pyplot as plt
import dask

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
        lock=None,
        masked=False,
        mask_and_scale=True,
        variable=None,
        group=None,
        default_name="band_data",
        decode_times=True,
        decode_timedelta=None,
        open_kwargs=None,
        dtype=np.int64
    ):
        size = np.dtype(dtype).itemsize

        shape  = filename_or_obj.shape
    
        backend_array = OpenVisusBackendArray(
            filename_or_obj=filename_or_obj, 
            shape=(shape), 
            dtype=dtype,
            lock=dask.utils.SerializableLock()

        )
        data = LazilyIndexedArray(backend_array)

        var = xr.Variable(dims=("x","y","z"), data=data)
        return xr.Dataset({"data_arr": var})

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext[1:].lower() in CAN_OPEN_EXTS

# subclass implementation of BackendArray
class OpenVisusBackendArray(BackendArray):
    def __init__(
        self,
        filename_or_obj,
        shape,
        dtype,
        lock
    ):
        self.filename_or_obj = filename_or_obj
        self.shape = shape
        self.dtype = dtype
        self.lock = lock
    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._raw_indexing_method
        )

    def _raw_indexing_method(self, key):

        key0 = key[0] 
        size = np.dtype(self.dtype).itemsize
        

            
        with self.lock, open(self.filename_or_obj) as f:
            arr = np.fromfile(f, np.int6t)
        

        return arr


BACKEND_ENTRYPOINTS["openvisus"]=OpenVisusEntryPoint

db=LoadDataset("http://atlantis.sci.utah.edu/mod_visus?dataset=BlueMarble")
print(db)

# def ShowData(data):
#     fig = plt.figure(figsize = (70,20))
#     ax = fig.add_subplot(1,1,1)
#     ax.imshow(data, origin='lower')
#     plt.show()

max_resolution=21
# logic_box=db.getLogicBox()
data=db.read(time=11,max_resolution=max_resolution)

# ShowData(data)
# print(data)

da=xr.open_dataarray(data, engine=OpenVisusEntryPoint, chunks={"x":100,"y":100,"z":3})


print(da)