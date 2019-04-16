import numpy as np
import xarray as xr
import pandas as pd
import Ngl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cftime
import os
import re

maindir = 'input/ymonmeans/'
inpfsuf = "all_mon.nc"
# inpfname = "input/ymonmeans/rcp2.6_seg_005/all_mon.nc"

inpvname = "TREFHT"

shpfname = "/home/gabriel/shapefiles/estados_2010.shp"

pdffname = "test.pdf"

maxlat = 5.0
minlat = -33.0
maxlon = 326.0
minlon = 280.0

runnames = os.listdir(maindir)
inpfnames = {maindir + '/' + i + '/' + inpfsuf : i for i in runnames}

# dicvars = {inpfname:xr.open_dataset(inpfname)[inpvname].sel(lat = slice(minlat,maxlat), lon = slice(minlon,maxlon)) for inpfname in inpfnames}
dicvars = {inpfname:xr.open_dataset(inpfname)[inpvname].sel(lat = slice(minlat,maxlat), lon = slice(minlon,maxlon)) for inpfname in inpfnames.keys()}
# Make sure the time axis are equal to the one in ref

ref = dicvars[list(inpfnames.keys())[0]].coords['time']
for inpfname in inpfnames.keys():
    dicvars[inpfname].coords['time'] = ref
    dicvars[inpfname] = dicvars[inpfname].expand_dims()
v = vars[0]
v = v.expand_dims('scenario')
v.coords['scenario'] = pd.Index(['poi'])
v.assign_coords(scenario = ('poi'))
v.set_coords()

n = runnames[0]
[re.split('_',n)[0:1] for n in runnames]

varall = xr.concat(vars, pd.Index(runnames, name='run'))

plot = xr.plot.imshow(varall[:,0,:,:], col = 'run', col_wrap=4, levels = range(290,310,2), cmap = "jet")
