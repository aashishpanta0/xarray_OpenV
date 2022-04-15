import xarray as xr
import numpy  as np
import pandas as pd


# !pip install OpenVisusNoGui
import OpenVisus as ov

# see https://xarray.pydata.org/en/stable/internals/how-to-add-new-backend.html


# ////////////////////////////////////////////////////////////
class OpenVisusBackendArray(xr.backends.common.BackendArray):
#     TODO: add num_refinements,quality
#     TODO: adding it for normalized coordinates

    # constructor
    def __init__(self,db, shape, dtype, timesteps,resolution,ncomponents,bool_coords):
        self.db    = db
        self.shape = shape
        self.dtype = dtype
        self.bool_coords=bool_coords
        self.ncomponents=ncomponents
        self.pdim=db.getPointDim()
        self.timesteps=timesteps
        self.resolution=resolution

    # _getKeyRange
    def _getKeyRange(self, value):
        A = value.start if isinstance(value, slice) else value    ; A = 0             if A is None else A
        B = value.stop  if isinstance(value, slice) else value + 1; B = np.max(self.shape) if B is None else B
        return (A,B)
    # __readSamples
    def _raw_indexing_method(self, key: tuple) -> np.typing.ArrayLike:
        print("_raw_indexing_method","key",key)
        if self.pdim==2:
        
            # getting selection from the data
            t1,t2=self._getKeyRange(key[0])
            y1,y2=self._getKeyRange(key[1])
            x1,x2=self._getKeyRange(key[2])
            c1,c2=self._getKeyRange(key[3])
            
            if isinstance(self.resolution,int):
                res=self.resolution
            else:
                res,res1=self._getKeyRange(key[-1])

                if res==0:
                    
                    res= self.db.getMaxResolution()
                    print('Using Max Resolution: ',res)
            if isinstance(self.timesteps,int):
                data=self.db.read(time=self.timesteps,max_resolution=res, logic_box=[(x1,y1),(x2,y2)])
            else:
                
                if isinstance(t1,int) and isinstance(res,int) and self.bool_coords==False:

                    data=self.db.read(time=t1,max_resolution=res,logic_box=[(x1,y1),(x2,y2)])
                else:
                    data=self.db.read(logic_box=[(x1,y1),(x2,y2)],max_resolution=self.db.getMaxResolution())
                
        elif self.pdim==3:
            
            t1,t2=self._getKeyRange(key[0])
            z1,z2=self._getKeyRange(key[1])
            y1,y2=self._getKeyRange(key[2])
            
            
            if isinstance(self.resolution,int):
                res=self.resolution
            else:
                res,res1=self._getKeyRange(key[-1])

                if res==0:
                    self.shape.pop()
                    res= self.db.getMaxResolution()
                    print('Using Max Resolution: ',res)
            if isinstance(self.timesteps,int):
                x1,x2=self._getKeyRange(key[3])
                data=self.db.read(time=self.timesteps,max_resolution=res, logic_box=[(x1,y1,z1),(x2,y2,z2)])
            elif len(self.timesteps)==1 and self.bool_coords==False:
                x1,x2=self._getKeyRange(key[3])
                data=self.db.read(max_resolution=res,logic_box=[(x1,y1,z1),(x2,y2,z2)])
            elif len(self.timesteps)==1 and self.bool_coords==True:
                data=self.db.read(logic_box=[(y1,z1,t1),(y2,z2,t2)])
 
            else:
                
                if isinstance(t1, int) and isinstance(res,int) and self.bool_coords==False:
                    x1,x2=self._getKeyRange(key[3])

                    data=self.db.read(time=t1, max_resolution=res,logic_box=[(x1,y1,z1),(x2,y2,z2)])
                elif isinstance(t1, int) and isinstance(res,int) and self.bool_coords==True:

                    data=self.db.read(logic_box=[(y1,z1,t1),(y2,z2,t2)])
                else:
                    data=self.db.read(logic_box=[(x1,y1,z1),(x2,y2,z2)])
                    
                    
        else:
            raise Exception("dimension error")

#         # last key element is the channel
        
        if self.ncomponents>1 :
             #Y,X,Channel
            if self.pdim==2:
                data=data[:,:,c1:c2]


            # Z,Y,X,Channel
            elif self.pdim==3:
                c1,c2=self._getKeyRange(key[4])

                data=data[:,:,:,c1:c2]

            else:
                raise Exception("problem")
        
        return np.array(data)
    # __getitem__
    def __getitem__(self, key: xr.core.indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        return xr.core.indexing.explicit_indexing_adapter(key,self.shape,
                                                          xr.core.indexing.IndexingSupport.BASIC,
                                                          self._raw_indexing_method)


class OpenVisusBackendEntrypoint(xr.backends.common.BackendEntrypoint):

    open_dataset_parameters = ["filename_or_obj", "drop_variables", "resolution", "timesteps","coords","attrs","dims"]
    
    # open_dataset
    def open_dataset(self,filename_or_obj,*, resolution=None, timesteps=None,drop_variables=None,coords=None,attrs=None,dims=None):

        self.resolution=resolution

        data_vars={}
        self.coordinates=coords
        self.attributes=attrs
        self.dimensions=dims

        db=ov.LoadDataset(filename_or_obj)
#         print(db.getMaxResolution())
        self.timesteps=timesteps
        dim=db.getPointDim()
        
        dims=db.getLogicSize()
        if self.timesteps==None:
            self.timesteps=[int(it) for it in db.getTimesteps().asVector()]
            
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
            shape=list(reversed(dims))
            bool_coords=False
            
            if self.coordinates==None:
            
            
                if dim==2:
                    labels=["y", "x"]
                elif dim==3:
                    labels=["z", "y", "x"]
                else:
                    raise Exception("assigning labels error")
                
#                shape=list(reversed(dims))
            
                if ncomponents>1:
                    labels.append("channel")
                    shape.append(ncomponents)
                labels.insert(0,"time")
                labels.append("resolution")
                if isinstance(self.resolution,int):
                    
                    shape.append(self.resolution+1)
                else:
                    shape.append(db.getMaxResolution()+1)
                if isinstance(self.timesteps, int):
                    shape.insert(0,self.timesteps+1)
                else:
                    shape.insert(0,len(self.timesteps))
            else:
                bool_coords=True
                labels=[i for i in self.dimensions]


            data_vars[fieldname]=xr.Variable(
                labels,
                xr.core.indexing.LazilyIndexedArray(OpenVisusBackendArray(db=db, shape=shape,dtype=dtype,
                                                                          timesteps=self.timesteps,
                                                                          resolution=self.resolution,
                                                                          ncomponents=ncomponents,
                                                                          bool_coords=bool_coords)),
                attrs={} # no attributes
            )

            print("Adding field ",fieldname,"shape ",shape,"dtype ",dtype,"labels ",labels,"timesteps ",self.timesteps,
                 "Max Resolution ", db.getMaxResolution())
        ds = xr.Dataset(data_vars=data_vars,coords=self.coordinates, attrs=dict(self.attributes))
        

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

        # todo: extend to S3 datasets
        if "mod_visus" in filename_or_obj:
            return True
         
        # local files
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext.lower()==".idx"


