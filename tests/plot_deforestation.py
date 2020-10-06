#%%
import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats
# import Ngl
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

# from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cftime
import os
import re
import copy
#%%

# maindir = '../input/out_surfdata/'
maindir = '../input/out_surfdata_fix/'
inpfpref = "surfdata.pftdyn_0.9x1.25_"
# p1suf = 'mon_avg_2013_2031.nc'
# p2suf = 'mon_avg_2032_2050.nc'
# inpfname = "input/ymonmeans/rcp2.6_seg_005/all_mon.nc"

dicfnames = {
    'rcp2.6':'surfdata.pftdyn_0.9x1.25_rcp2.6_simyr1850-2100_c100323.nc',
    'rcp8.5':'surfdata.pftdyn_0.9x1.25_rcp8.5_simyr1850-2100_c100319.nc',
    'seg':'surfdata.pftdyn_0.9x1.25_rcp2.6_simyr1850-2100_rochedo_seg_c100323.nc',
    'weg':'surfdata.pftdyn_0.9x1.25_rcp2.6_simyr1850-2100_rochedo_weg_c100323.nc',
    'veg':'surfdata.pftdyn_0.9x1.25_rcp2.6_simyr1850-2100_veg_c100323.nc'
    }

inpvname = "PCT_PFT"

shpfname = "/home/gabriel/shapefiles/estados_2010.shp"

pdffname = "tests/landuse.pdf"

years = [2005,2030,2050]

maxlat = 5.0
minlat = -33.0
maxlon = 326.0
minlon = 280.0

#%%
allarrays = []
for scenario in dicfnames:
    fname = maindir + dicfnames[scenario]
    arr = xr.open_dataset(fname)[inpvname]
    # arr['lsmlat'] = ((xr.open_dataset(fname)['LATN'] + xr.open_dataset(fname)['LATS'])/2.0).mean(dim = 'lsmlon')
    # arr['lsmlon'] = ((xr.open_dataset(fname)['LONE'] + xr.open_dataset(fname)['LONW'])/2.0).mean(dim = 'lsmlat')
    # arr = arr.rename(dict(zip(list(arr.dims),[i[3:6] if i[0:3] == 'lsm' else i for i in list(arr.dims)])))
    # arr = arr.sel(time = years, lat = slice(minlat,maxlat), lon = slice(minlon,maxlon))
    arr.name = scenario
    allarrays.append(arr)

#%%
# alldeffrac = {item.name:item.isel(pft = [14,15,16]).sum(dim = "pft").sel(time = years) for item in allarrays}
alldeffrac = [item.isel(pft = [14,15,16]).sum(dim = "pft").sel(time = years) for item in allarrays]

# for arr in alldeffrac:
for i in range(len(alldeffrac)):
    arr = alldeffrac[i]
    arr = arr.expand_dims("scenario")
    arr.coords['scenario'] = pd.Index([arr.name])
    arr.name = "DEF"

alldeffrac[0]
# xr.merge(alldeffrac)
#%%
# allarrays = []
# for scenario in dicfnames:
#     fname = maindir + dicfnames[scenario]
#     arr = xr.open_dataset(fname)[inpvname]
#     arr['lsmlat'] = ((xr.open_dataset(fname)['LATN'] + xr.open_dataset(fname)['LATS'])/2.0).mean(dim = 'lsmlon')
#     arr['lsmlon'] = ((xr.open_dataset(fname)['LONE'] + xr.open_dataset(fname)['LONW'])/2.0).mean(dim = 'lsmlat')
#     arr = arr.rename(dict(zip(list(arr.dims),[i[3:6] if i[0:3] == 'lsm' else i for i in list(arr.dims)])))
#     arr = arr.sel(time = years, lat = slice(minlat,maxlat), lon = slice(minlon,maxlon))
#     arr.name = scenario
#     allarrays.append(arr)



