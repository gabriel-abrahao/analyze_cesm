#%% Python package that makes working with labelled multi-dimensional arrays simple and efficient
import Ngl
import os
import glob
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
%matplotlib inline

#%%
# filename = "/media/gabriel/hd2_6tb/ccsm4/test/rcp0.0_veg_005.cam2.h0.1950-01.nc"

infolder = "/media/gabriel/hd2_6tb/sdumont/monthly_sdumont/rcp0.0_veg_005/run/"
filenames = glob.glob(infolder + '*1950*')

varname = "T"

# ds = xr.open_dataset(filename)
ds = xr.open_mfdataset(glob.glob(infolder + '*1950*'))
#%%



#  Extract the desired variables (need numpy arrays for vertical interpolation)
hyam = ds["hyam"].isel(time = 0)
hybm = ds["hybm"].isel(time = 0)
# psrf = ds["PS"].isel(time = 0)
psrf = ds["PS"]
P0mb =  0.01*ds["P0"].isel(time = 0)

# hyam = ds["hyam"]
# hybm = ds["hybm"]
# psrf = ds["PS"]
# P0mb =  0.01*ds["P0"]


var    = ds[varname]

lats = ds["lat"]
lons = ds["lon"]
times = ds['time']


#  Define the output pressure levels.
# pnew = [850.]
pnew = [1000.0,925.0,850.0,700.0,600.0,500.0,400.0,300.0,250.0,200.0,150.0,100.0,70.0,50.0,30.0,20.0,10.0]
lev = xr.DataArray(pnew, coords = [pnew], dims = "lev", attrs = {"units" : "hPa"})


#  Do the interpolation.
intyp = 1                              # 1=linear, 2=log, 3=log-log
kxtrp = False                          # True=extrapolate (when the output pressure level is outside of the range of psrf)

# FIXME:Load the variable onto memory if it's a dask array. 
# Not sure if Ngl works well with those, check later
if var.chunks is not None:
    var.load()
    hyam.load()
    hybm.load()
    psrf.load()
    P0mb.load()
    lats.load()
    lons.load()
    times.load()


#%%
# Vertical interpolation
varnew = Ngl.vinth2p(var,hyam,hybm,pnew,psrf,intyp,P0mb,1,kxtrp)
# varnew = Ngl.vinth2p(var3,hyam.values,hybm.values,pnew,psrf.values,intyp,P0mb.values,1,kxtrp)
# varnew = Ngl.vinth2p(var.isel(time = [0]).values,hyam,hybm,pnew,psrf,intyp,P0mb,1,kxtrp)

# varnew[varnew==1e30] = np.NaN
varnew.shape
# varnew

#%%
# Create new xarray Dataset

dsnew = xr.Dataset({varname: (('time','lev','lat','lon'), varnew[:,:,:,:])},
                  {'lon':  lons, 'lat':  lats, 'time': times, 'lev' : lev})
# dsnew = xr.Dataset({varname: (('time','lev','lat','lon'), varnew[:,:,:,:])},
                #   {'lon':  lons, 'lat':  lats, 'time': [0,1], 'lev' : lev})

dsnew["lev"] = dsnew["lev"]*100.0
dsnew["lev"].attrs['units'] = 'Pa'

dsnew[varname].name = ds[varname].name
dsnew[varname].attrs = ds[varname].attrs


dsnew

#%%
dsnew.to_netcdf("~/ccsm4/test/lala.nc")

#%%
maxlat = 5.0
minlat = -33.0
maxlon = 326.0
# minlon = 280.0
minlon = 300.0

pds = dsnew.sel(lat = slice(minlat,maxlat), lon = slice(minlon,maxlon))

# pds[varname].sel(lev = 925).plot()
pds[varname].isel(time=0).sel(lat = [-20,-10,0], lon = 300, method = "nearest").plot.line(x = "lev")


#%%
# Plotting
fig = plt.figure(figsize=(15, 5))
ax = plt.axes(projection=ccrs.PlateCarree())

dsnew.T850.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=load_cmap('vik'))

ax.coastlines()
plt.title(ds.time.values[0].strftime("%B year %Y"))