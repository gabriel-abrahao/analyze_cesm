# STEP1 of preprocessing raw 3D files
# Choose a time period and calculate monthly averages of
# each scenario member to a climatological monthly file
#%% 
import Ngl
import os
import re
import glob
from datetime import datetime
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
# %matplotlib inline

#%%
# filename = "/media/gabriel/hd2_6tb/ccsm4/test/rcp0.0_veg_005.cam2.h0.1950-01.nc"

basefolder = "/media/gabriel/hd2_6tb/sdumont/monthly_sdumont/"
baseoutfolder = "output/"

scens = ["rcp8.5_weg", "rcp8.5_seg", "rcp2.6_weg", "rcp2.6_seg"]

syear = 2040
eyear = 2050

years = list(range(syear, eyear + 1))

#%% Scenario loop
# scen = scens[0]
for scen in scens:

    # Create output folder
    outfolder = baseoutfolder + scen + "/"
    os.makedirs(outfolder, exist_ok=True)

    regex = re.compile(scen + "_.*")
    infolders = [basefolder + i + "/run/" for i in os.listdir(basefolder) if regex.match(i) ]
    #%% Folders (members) loop
    # infolder = infolders[0]
    for infolder in infolders:

        memberstr = (os.path.split(os.path.split(os.path.split(infolder)[0])[0])[1]).split(sep = "_")[-1]

        # filenames = glob.glob(infolder + '*{2030..2050}*')
        filenames = [glob.glob(infolder + "*." + mon + ".*")  for mon in [str(year) + "-" + i for year in years for i in [str(j).zfill(2) for j in range(1,13)]]]
        filenames = [i for j in filenames for i in j] # Flatten it

        #%% Open a folder's (member) dataset
        inds = xr.open_mfdataset(filenames)
        tmeands = inds.groupby("time.month").mean(keep_attrs = True)

        #%% Save output. Since we're lazy loading, this is where most of the processing happens
        outfname = outfolder + scen + "_" + memberstr + "_" + str(syear) + "_" + str(eyear) + ".nc"
        print(outfname + " | " + datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        tmeands.to_netcdf(outfname)


#%%
# This was for the vertical interpolation, we'll do it in a later step
# # ds = xr.open_dataset(filename)
# ds = xr.open_mfdataset(glob.glob(infolder + '*1950*'))
# #%%
# varname = "T"

# #  Extract the desired variables (need numpy arrays for vertical interpolation)
# hyam = ds["hyam"].isel(time = 0)
# hybm = ds["hybm"].isel(time = 0)
# # psrf = ds["PS"].isel(time = 0)
# psrf = ds["PS"]
# P0mb =  0.01*ds["P0"].isel(time = 0)

# # hyam = ds["hyam"]
# # hybm = ds["hybm"]
# # psrf = ds["PS"]
# # P0mb =  0.01*ds["P0"]


# var    = ds[varname]

# lats = ds["lat"]
# lons = ds["lon"]
# times = ds['time']


# # Define the output pressure levels. 
# # We need them in hPa/mb here for Ngl.vinth2p, but convert to Pa later for consistency with reference CMIP5 files
# pnew = [1000.0,925.0,850.0,700.0,600.0,500.0,400.0,300.0,250.0,200.0,150.0,100.0,70.0,50.0,30.0,20.0,10.0]
# lev = xr.DataArray(pnew, coords = [pnew], dims = "lev", attrs = {"units" : "hPa"})


# #  Do the interpolation.
# intyp = 1                              # 1=linear, 2=log, 3=log-log
# kxtrp = True                          # True=extrapolate (when the output pressure level is outside of the range of psrf)

# # FIXME:Load the variable onto memory if it's a dask array. 
# # Not sure if Ngl works well with those, check later
# # if var.chunks is not None:
# #     var.load()
# #     hyam.load()
# #     hybm.load()
# #     psrf.load()
# #     P0mb.load()
# #     lats.load()
# #     lons.load()
# #     times.load()


# #%%
# # Vertical interpolation
# varnew = Ngl.vinth2p(var,hyam,hybm,pnew,psrf,intyp,P0mb,1,kxtrp)
# # varnew = Ngl.vinth2p(var3,hyam.values,hybm.values,pnew,psrf.values,intyp,P0mb.values,1,kxtrp)
# # varnew = Ngl.vinth2p(var.isel(time = [0]).values,hyam,hybm,pnew,psrf,intyp,P0mb,1,kxtrp)

# # varnew[varnew==1e30] = np.NaN
# varnew.shape
# # varnew

# #%%
# # Create new xarray Dataset

# dsnew = xr.Dataset({varname: (('time','lev','lat','lon'), varnew[:,:,:,:])},
#                   {'lon':  lons, 'lat':  lats, 'time': times, 'lev' : lev})
# # dsnew = xr.Dataset({varname: (('time','lev','lat','lon'), varnew[:,:,:,:])},
#                 #   {'lon':  lons, 'lat':  lats, 'time': [0,1], 'lev' : lev})

# dsnew["lev"] = dsnew["lev"]*100.0
# dsnew["lev"].attrs['units'] = 'Pa'

# dsnew[varname].name = ds[varname].name
# dsnew[varname].attrs = ds[varname].attrs


# dsnew

# #%%
# dsnew.to_netcdf("~/ccsm4/test/lala.nc")

# #%%
# dsnew.rename(T = "ta").to_netcdf("~/ccsm4/test/lala.nc")

# #%%
# maxlat = 5.0
# minlat = -33.0
# maxlon = 326.0
# # minlon = 280.0
# minlon = 300.0

# pds = dsnew.sel(lat = slice(minlat,maxlat), lon = slice(minlon,maxlon))

# # pds[varname].sel(lev = 925).plot()
# pds[varname].isel(time=0).sel(lat = [-20,-10,0], lon = 300, method = "nearest").plot.line(x = "lev")


# #%%
# # Plotting
# fig = plt.figure(figsize=(15, 5))
# ax = plt.axes(projection=ccrs.PlateCarree())

# dsnew.T850.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=load_cmap('vik'))

# ax.coastlines()
# plt.title(ds.time.values[0].strftime("%B year %Y"))