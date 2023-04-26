import xarray as xr
import numpy  as np
import pandas as pd
import pickle

# !pip install OpenVisusNoGui
import OpenVisus as ov

# see https://xarray.pydata.org/en/stable/internals/how-to-add-new-backend.html


# ////////////////////////////////////////////////////////////
class OpenVisusBackendArray(xr.backends.common.BackendArray):
#     TODO: add num_refinements,quality
#     TODO: adding it for normalized coordinates

    # constructor
    def __init__(self,db, shape, dtype, field,timesteps,resolution,ncomponents,bool_coords):
        self.db    = db
        self.shape = shape
        self.dtype = dtype
        self.field=field
        self.bool_coords=bool_coords
        self.ncomponents=ncomponents
        self.pdim=db.getPointDim()
        self.timesteps=timesteps
        self.resolution=resolution

    # _getKeyRange
    def _getXRange(self, value):
        if self.pdim==2:
            A = value.start if isinstance(value, slice) else value    ; A = int(0)             if A is None else A
            B = value.stop  if isinstance(value, slice) else value + 1; B = int(self.shape[2]) if B is None else B
        if self.pdim==3:
            A = value.start if isinstance(value, slice) else value    ; A = int(0)             if A is None else A
            B = value.stop  if isinstance(value, slice) else value + 1; B = int(self.shape[3]) if B is None else B
        return (A,B)
    def _getYRange(self, value):
        if self.pdim==2:
            A = value.start if isinstance(value, slice) else value    ; A = 0             if A is None else A
            B = value.stop  if isinstance(value, slice) else value + 1; B = int(self.shape[1]) if B is None else B
        if self.pdim==3:
            A = value.start if isinstance(value, slice) else value    ; A = int(0)             if A is None else A
            B = value.stop  if isinstance(value, slice) else value + 1; B = int(self.shape[2]) if B is None else B
        return (A,B)

    def _getKeyRange(self, value):
        A = value.start if isinstance(value, slice) else value    ; A = 0             if A is None else A
        B = value.stop  if isinstance(value, slice) else value + 1; B = np.max(self.shape) if B is None else B
        return (A,B)
    
    def _getTRange(self, value):

        A =  value.start if isinstance(value, slice) else value    ;A= int(self.shape[0])-1 if A is None else A
        B =  value.stop  if isinstance(value, slice) else value + 1; B=1 if B is None else B

        return (A,B)
    
    # __readSamples
    def _raw_indexing_method(self, key: tuple) -> np.typing.ArrayLike:
        print("_raw_indexing_method","key",key)
        if self.pdim==2:
            # getting selection from the data
            t1,t2=self._getTRange(key[0])
            x1,x2=self._getXRange(key[2])
            y1,y2=self._getYRange(key[1])
            c1,c2=self._getKeyRange(key[3])
            if isinstance(self.resolution,int):
                
                res=self.resolution
            else:
                res,res1=self._getKeyRange(key[-1])
                print(res)
                if res==0:
                    res= self.db.getMaxResolution()
                    print('Using Max Resolution: ',res)
            if isinstance(self.timesteps,int):
                data=self.db.read(time=self.timesteps,max_resolution=res, logic_box=[(x1,y1),(x2,y2)],field=self.field)
            else:
                data=self.db.read(time=t1,max_resolution=res,logic_box=[(x1,y1),(x2,y2)],field=self.field)
                # else:
                #     data=self.db.read(logic_box=[(x1,y1),(x2,y2)],max_resolution=self.db.getMaxResolution())
                
        elif self.pdim==3:
            print('3D')
            
            t1,t2=self._getKeyRange(key[0])
            z1,z2=self._getKeyRange(key[1])
            y1,y2=self._getYRange(key[2])
             
            if isinstance(self.resolution,int):
                res=self.resolution
            else:
                res,res1=self._getKeyRange(key[-1])
                print('here')

                if res==0:
                    print('hereee')
                    self.shape.pop()
                    res= self.db.getMaxResolution()
                    print('Using Max Resolution: ',res)
            if isinstance(self.timesteps,int):
                x1,x2=self._getXRange(key[3])
                data=self.db.read(time=self.timesteps,max_resolution=res, logic_box=[(x1,y1,z1),(x2,y2,z2)],field=self.field)
            elif len(self.timesteps)==1 and self.bool_coords==False:
                x1,x2=self._getXRange(key[3])
                data=self.db.read(max_resolution=res,logic_box=[(x1,y1,z1),(x2,y2,z2)],field=self.field)
            elif len(self.timesteps)==1 and self.bool_coords==True:
                data=self.db.read(logic_box=[(y1,z1,t1),(y2,z2,t2)])
 
            else:
                
                if isinstance(t1, int) and isinstance(res,int) and self.bool_coords==False:
                    x1,x2=self._getXRange(key[3])

                    data=self.db.read(time=t1, max_resolution=res,logic_box=[(x1,y1,z1),(x2,y2,z2)],field=self.field)
                elif isinstance(t1, int) and isinstance(res,int) and self.bool_coords==True:

                    data=self.db.read(logic_box=[(y1,z1,t1),(y2,z2,t2)],field=self.field)
                else:
                    data=self.db.read(logic_box=[(x1,y1,z1),(x2,y2,z2)],field=self.field)
                    
                    
        else:
            raise Exception("dimension error")

        print(data.shape)
        
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

    open_dataset_parameters = ["filename_or_obj", "drop_variables", "resolution", "timesteps","coords","da_attrs","ds_attrs","dims"]
    
    # open_dataset
    def open_dataset(self,filename_or_obj,*, resolution=None, timesteps=None,drop_variables=None,coords=None,da_attrs=None,ds_attrs=None,dims=None):

        self.resolution=resolution

        data_vars={}
        self.ds_attrs=ds_attrs
        self.da_attrs=da_attrs
        self.dimensions=dims
        self.coords=coords
        # print(dims)

        db=ov.LoadDataset(filename_or_obj)
#         print(db.getMaxResolution())
        self.timesteps=timesteps
        dim=db.getPointDim()
        dims=db.getLogicSize()
        if self.timesteps==None:
            self.timesteps=db.getTimesteps()
            
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
        
            if True:
                if dim==2:
                    labels=["x", "y"]
                elif dim==3:
                    labels=["x", "y", "z"]
                else:
                    raise Exception("assigning labels error")
                
            #    shape=list(reversed(dims))
            
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
                                                                          field=field,
                                                                          timesteps=self.timesteps,
                                                                          resolution=self.resolution,
                                                                          ncomponents=ncomponents,
                                                                          bool_coords=bool_coords))
                # no attributes
            )

            print("Adding field ",fieldname,"shape ",shape,"dtype ",dtype,"labels ",labels,"timesteps ",self.timesteps,
                 "Max Resolution ", db.getMaxResolution())
            if da_attrs!=None and fieldname in da_attrs:
                attr=da_attrs[fieldname]
                for keys in attr:
                    data_vars[fieldname].attrs[keys]=attr[keys]
                
            
        ds = xr.Dataset(data_vars=data_vars,coords=coords)
        if ds_attrs!=None:
            ds.assign_attrs(ds_attrs)

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


def save_pickle(values,filename):
    with open(filename,'wb') as fb:
        pickle.dump(values,fb)
    print('json saved locally')
    return
        
def load_pickle(filename):
    with open(filename,'rb') as fb:
        data=pickle.load(fb)
    return data
    
def open_dataset(filename):
    idx_file=filename[:-3]
    idx_file=idx_file+'.idx'
    print('Loading IDX file: '+str(idx_file))
    d=xr.open_dataset(filename)
    ds=xr.Dataset()
    db=ov.LoadDataset(idx_file)
    for f in db.getFields():
        data=db.read(field=f)
        ds[f]=xr.DataArray(data,dims=d[f].dims,coords=d[f].coords,attrs=d[f].attrs)
        
    return ds
