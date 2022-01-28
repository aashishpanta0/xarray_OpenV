from xarray_files import *
import matplotlib.pyplot as plt


def ShowData(data):
    fig = plt.figure(figsize = (70,20))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(data, origin='lower')
    plt.show()

"""
For now, the dataset is loaded at full resolution
"""
ds = xr.open_dataset("http://atlantis.sci.utah.edu/mod_visus?dataset=BlueMarble",engine=OpenVisusBackendEntrypoint)

print("dtype",ds.data.dtype)
print("shape",ds.data.shape)
#print("first-sample",ds.data[0,0,:].values)
print("subregion",ds.data[0:3,1:3,:].values)



ShowData(ds.data[1000:10000,1000:10000,:].values)


db=xr.open_dataset('http://atlantis.sci.utah.edu/mod_visus?dataset=david_subsampled', engine=OpenVisusBackendEntrypoint)
"""
Using sel() method from Xarray on OpenVisus to slice up the data and select certain certain range values of x and y
"""
# ShowData(db.data[10000:11000,10000:11000,:].values)
ShowData(db.sel(x=slice(10000,11000),y=slice(10000,11000),channel=slice(0,3)).data.values)


db2=xr.open_dataset('http://atlantis.sci.utah.edu/mod_visus?dataset=2kbit1', engine=OpenVisusBackendEntrypoint)
select_db2=db2.sel(x=slice(0,2048),y=slice(0,2048),z=slice(1024,1025)).to_array()

ShowData(select_db2[0][0].values)

db=xr.open_dataset('http://atlantis.sci.utah.edu/mod_visus?dataset=david_subsampled', engine=OpenVisusBackendEntrypoint)
# Doesn't work for normalized coordinates yet

# ###ShowData(db.data[0.35:0.45,0.8:0.9,:].values)