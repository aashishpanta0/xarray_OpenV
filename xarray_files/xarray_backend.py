from xarray.backends.common import BACKEND_ENTRYPOINTS
import xarray as xr
import numpy  as np
import pandas as pd
import os
# !pip install OpenVisusNoGui
import OpenVisus as ov

# see https://xarray.pydata.org/en/stable/internals/how-to-add-new-backend.html


# ////////////////////////////////////////////////////////////
class OpenVisusBackendArray(xr.backends.common.BackendArray):

    # constructor
    def __init__(self,db, shape, dtype, ncomponents):
        self.db    = db
        self.shape = shape
        self.dtype = dtype
        self.ncomponents=ncomponents
        self.pdim=db.getPointDim()

    # __getitem__
    def __getitem__(self, key: xr.core.indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        return xr.core.indexing.explicit_indexing_adapter(key,self.shape,xr.core.indexing.IndexingSupport.BASIC,self._raw_indexing_method)

    # _getKeyRange
    def _getKeyRange(self, value):
        A = value.start if isinstance(value, slice) else value    ; A = 0             if A is None else A
        B = value.stop  if isinstance(value, slice) else value + 1; B = self.shape[1] if B is None else B
        return (A,B)

    # __readSamples
    def _raw_indexing_method(self, key: tuple) -> np.typing.ArrayLike:
        print("_raw_indexing_method","key",key)

        if self.pdim==2:
            y1,y2=self._getKeyRange(key[0])
            x1,x2=self._getKeyRange(key[1])
            data=self.db.read(logic_box=[(x1,y1),(x2,y2)])

        elif self.pdim==3:
            z1,z2=self._getKeyRange(key[0])
            y1,y2=self._getKeyRange(key[1])
            x1,x2=self._getKeyRange(key[2])
            data=self.db.read(logic_box=[(x1,y1,z1),(x2,y2,z2)])

        else:
            raise Exception("problem")

        # last key element is the channel
        if self.ncomponents>1:
            c1,c2=self._getKeyRange(key[-1])

            # Y,X,Channel
            if self.pdim==2:
                data=data[:,:,c1:c2]

            # Z,Y,X,Channel
            elif self.pdim==3:
                data=data[:,:,:,c1:c2]

            else:
                raise Exception("problem")

        return data




# ////////////////////////////////////////////////////////////
class OpenVisusBackendEntrypoint(xr.backends.common.BackendEntrypoint):

    open_dataset_parameters = ["filename_or_obj", "drop_variables", "resolution", "timesteps"]
    
    # open_dataset
    def open_dataset(self,filename_or_obj,*, resolution=None, timesteps=None, drop_variables=None):

        # TODO for now only full resoluton
        self.resolution=resolution

        # TODO for now only default timestep
        self.timesteps=timesteps

        data_vars={}

        db=ov.LoadDataset(filename_or_obj)

        dim=db.getPointDim()
        dims=db.getLogicSize()

#         print(db.getDatasetBody().toString())
#         print("dim",dim, "dims",dims)

        # convert OpenVisus fields into xarray variables
        for fieldname in db.getFields():
            field=db.getField(fieldname)
            
            ncomponents=field.dtype.ncomponents()
            atomic_dtype=field.dtype.get(0)

            # note: OpenVisus can have dtype uint8[3],numpy cannot
            # for this reason for example
            # 1024x768     uint8       becomes (768,1024)      np.uint8    (NOTE I don't have to add anything to the shape)
            # 1024x768     uint8[3]    becomes (768,1024,3)    np.uint8    (NOTE I am adding a '3' to the shape)
            # 100x200x300  float64     becomes (300,200,100)   np.float64  (NOTE I don't have to add anything to the shape)
            # 100x200x300  float64[2]  becomes (300,200,100,2) np.float64  (NOTE I am adding a '2' to the shape)

            dtype=self.toNumPyDType(atomic_dtype) 

            if dim==2:
                labels=["y", "x"]
            elif dim==3:
                labels=["z", "y", "x"]
            else:
                raise Exception(todo)

            shape=list(reversed(dims))
            if ncomponents>1:
                labels.append("channel")
                shape.append(ncomponents)

            data_vars[fieldname]=xr.Variable(
                labels, 
                xr.core.indexing.LazilyIndexedArray(OpenVisusBackendArray(db=db, shape=shape,dtype=dtype, ncomponents=ncomponents)), 
                attrs={} # no attributes
            )

            print("Adding field",fieldname,"shape",shape,"dtype",dtype,"labels",labels)

        ds = xr.Dataset(data_vars=data_vars)
        ds.set_close(self.close_method)
        return ds
    
    # toNumPyDType (always pass the atomic OpenVisus type i.e. uint8[8] should not be accepted)
    def toNumPyDType(self,atomic_dtype):
        """
        convert an Openvisus dtype to numpy dtype
        """

        # dtype  (<: little-endian, >: big-endian, |: not-relevant) ; integer providing the number of bytes  ; i (integer) u (unsigned integer) f (floating point)
        return np.dtype("".join([
            "|" if atomic_dtype.getBitSize()==8 else "<",
            "f" if atomic_dtype.isDecimal() else ("u" if atomic_dtype.isUnsigned() else "i"),
            str(int(atomic_dtype.getBitSize()/8))
        ])) 

    # close_method
    def close_method(self):
        print("nothing to do here")
    
    # guess_can_open
    def guess_can_open(self, filename_or_obj):
        print("guess_can_open",filename_or_obj)

        # this are remote datasets
        # todo: extend to S3 datasets
        if "mod_visus" in filename_or_obj:
            return True
         
        # local files
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext.lower()==".idx"
BACKEND_ENTRYPOINTS["openvisus"]=OpenVisusBackendEntrypoint

