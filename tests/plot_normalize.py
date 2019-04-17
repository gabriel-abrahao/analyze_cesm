import numpy as np
import xarray as xr
import pandas as pd
import scipy
import Ngl
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cftime
import os
import re

maindir = 'input/ymonmeans/'
inpfsuf = "all_mon.nc"
p1suf = 'mon_avg_2013_2031.nc'
p2suf = 'mon_avg_2032_2050.nc'
# inpfname = "input/ymonmeans/rcp2.6_seg_005/all_mon.nc"

inpvname = "TREFHT"

shpfname = "/home/gabriel/shapefiles/estados_2010.shp"

pdffname = "test.pdf"

maxlat = 5.0
minlat = -33.0
maxlon = 326.0
minlon = 280.0

# runnames = os.listdir(maindir)
runnames = ["rcp8.5_seg_006"]
# inpfnames = {maindir + '/' + i + '/' + inpfsuf : i for i in runnames}
alldata = [{'p1fname': maindir + '/' + i + '/' + p1suf, 'p2fname': maindir + '/' + i + '/' + p2suf ,'runname' : i} for i in runnames]
for item in alldata:
    item['p1'] = xr.open_dataset(item['p1fname'])[inpvname].sel(lat = slice(minlat,maxlat), lon = slice(minlon,maxlon))
    item['p2'] = xr.open_dataset(item['p2fname'])[inpvname].sel(lat = slice(minlat,maxlat), lon = slice(minlon,maxlon))



# Make sure the time axis are equal to the one in ref
# ref = alldata[0]['p1'].coords['time']
# for inpfname in inpfnames.keys():
#     dicvars[inpfname].coords['time'] = ref
#     dicvars[inpfname] = dicvars[inpfname].expand_dims()
for item in alldata:
    sce, ens = item['runname'].rsplit('_',1)
    item['p1'] = item['p1'].expand_dims('scenario')
    item['p1'].coords['scenario'] = pd.Index([sce])
    item['p1'] = item['p1'].expand_dims('ensemble')
    item['p1'].coords['ensemble'] = pd.Index([ens])
    item['p2'] = item['p2'].expand_dims('scenario')
    item['p2'].coords['scenario'] = pd.Index([sce])
    item['p2'] = item['p2'].expand_dims('ensemble')
    item['p2'].coords['ensemble'] = pd.Index([ens])

allp1 = xr.merge([item['p1'] for item in alldata]).to_array().squeeze('variable')
allp2 = xr.merge([item['p2'] for item in alldata]).to_array().squeeze('variable')
# Math won't work if times are different
allp1.coords['time'] = allp2.coords['time']
anom = allp2-allp1
anomsig = scipy.stats.ttest_ind(allp2,allp1,axis=2) #Not quite right


poi = xr.DataArray(allp2.values - allp1.values, coords = allp2.coords)
poi

# plot = xr.plot.imshow(anom.isel(time = 0), col = 'scenario', row = 'ensemble', cmap = "jet")
# plot = xr.plot.imshow(anom.mean(dim = 'ensemble'), col = 'scenario', row = 'time', cmap = "jet")


v1 = allp1.isel(time = 0, scenario = 0, ensemble = 0)
sig = xr.where(v1 <= 290.0, 1.0, 0.0)

xr.plot.imshow(v1, cmap = "jet")
xr.plot.contourf(sig, levels = [0,0.99,2],hatches=['','.'],alpha = 0)

# plt.show()

# hvplot.quadmesh(anom)


# plot = xr.plot.imshow(allp1.isel(time = 0), col = 'scenario', row = 'ensemble', levels = np.arange(290.0,300.0,0.5), cmap = "jet")

# plot = xr.plot.imshow(varall[:,0,:,:], col = 'run', col_wrap=4, levels = range(290,310,2), cmap = "jet")
