#%% Python package that makes working with labelled multi-dimensional arrays simple and efficient
import Ngl
import os
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
%matplotlib inline

#%%
filename = "/media/gabriel/hd2_6tb/ccsm4/test/rcp0.0_veg_005.cam2.h0.1950-01.nc"

varname = "T"

ds = xr.open_dataset(filename)

#  Extract the desired variables (need numpy arrays for vertical interpolation)
hyam = ds["hyam"]
hybm = ds["hybm"]
T    = ds[varname]
psrf = ds["PS"]
P0mb =  0.01*ds["P0"]

lats = ds["lat"]
lons = ds["lon"]
times = ds['time']


#  Define the output pressure levels.
# pnew = [850.]
pnew = [1000.0,925.0,850.0,700.0,600.0,500.0,400.0,300.0,250.0,200.0,150.0,100.0,70.0,50.0,30.0,20.0,10.0]
lev = xr.DataArray(pnew, coords = [pnew], dims = "lev", attrs = {"units" : "hPa"})


#  Do the interpolation.
intyp = 1                              # 1=linear, 2=log, 3=log-log
kxtrp = True                          # True=extrapolate (when the output pressure level is outside of the range of psrf)

#%%
# Vertical interpolation

Tnew = Ngl.vinth2p(T,hyam,hybm,pnew,psrf,intyp,P0mb,1,kxtrp)

# Tnew[Tnew==1e30] = np.NaN
Tnew.shape
#%%
# Create new xarray Dataset

dset_p = xr.Dataset({varname: (('time','lev','lat','lon'), Tnew[:,:,:,:])},
                  {'lon':  lons, 'lat':  lats, 'time': times, 'lev' : lev})
dset_p[varname].attrs['units'] = 'K'
dset_p[varname].attrs['long_name'] = "Temperature"
dset_p[varname].attrs['standard_name'] = 'T'


#%%
maxlat = 5.0
minlat = -33.0
maxlon = 326.0
# minlon = 280.0
minlon = 300.0

pds = dset_p.sel(lat = slice(minlat,maxlat), lon = slice(minlon,maxlon))

# pds[varname].sel(lev = 925).plot()
pds[varname].isel(time=0).sel(lat = [-20,-10,0], lon = 300, method = "nearest").plot.line(x = "lev")


#%%
# Plotting
fig = plt.figure(figsize=(15, 5))
ax = plt.axes(projection=ccrs.PlateCarree())

dset_p.T850.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=load_cmap('vik'))

ax.coastlines()
plt.title(ds.time.values[0].strftime("%B year %Y"))