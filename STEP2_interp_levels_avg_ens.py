# STEP2 of preprocessing raw 3D files
# Calculate ensemble averages and interpolate
# hybrid sigma to pressure levels
#%% 
import Ngl
import os
import re
import glob
from datetime import datetime
import xarray as xr
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from numba import jit
# %matplotlib inline

from pooled_stats import *

#%%
# filename = "/media/gabriel/hd2_6tb/ccsm4/test/rcp0.0_veg_005.cam2.h0.1950-01.nc"

basefolder = "output/"
baseoutfolder = "output/"

scens = ["rcp8.5_weg", "rcp8.5_seg", "rcp2.6_weg", "rcp2.6_seg"]

syear = 2040
eyear = 2050
years = list(range(syear, eyear + 1))

nyears = len(years)

# Interpolation parameters
# New pressure levels. We need them in hPa/mb here for Ngl.vinth2p, 
# but will convert them to Pa later for consistency with reference CMIP5 files
pnew = [1000.0,925.0,850.0,700.0,600.0,500.0,400.0,300.0,250.0,200.0,150.0,100.0,70.0,50.0,30.0,20.0,10.0]
intyp = 1   # 1=linear, 2=log, 3=log-log
kxtrp = True    # Should we extrapolate into levels below the topography?

# %% Function definitions
def drop_unused_vars(ds):
    # smallvarnames = [var for var in ds.data_vars if ds[var].shape.__len__() < 4]
    fixedvarnames = [var for var in ds.data_vars if ds[var].shape.__len__() <= 2]
    smallvarnames = [\
        "CLDHGH", "CLDLOW", "CLDMED", "CLDTOT", \
        "FSDS", "FLDS", "SOLIN", "SRFRAD",\
        "LHFLX", "SHFLX", "QFLX",\
        "PBLH", "PHIS", "PS","PSL",
        "TMQ",\
        "PRECC", "PRECL", "PRECT",\
        "QREFHT", "RHREFHT", "TREFHT", "TREFMNAV", "TREFMXAV","TS","U10",\
        ]
    smallvarnames.extend([i + "_var" for i in smallvarnames])
    bigvarnames = ["Z3","T","Q","U","V","OMEGA","VQ","OMEGAQ"] #There is no UQ for some reason
    bigvarnames.extend([i + "_var" for i in bigvarnames])
    # bigvarnames = ["Z3","T"]
    bothvarnames = smallvarnames + bigvarnames + fixedvarnames
    dropvarnames = [i for i in ds.data_vars if i not in bothvarnames]
    ds = ds[bothvarnames]
    return(ds)

#%% Scenario loop
# scen = scens[3]
for scen in scens:
    print("=================================================")
    print("Running " + scen)
    print("=================================================")
    # Create output folder
    outfolder = baseoutfolder + scen + "/"
    os.makedirs(outfolder, exist_ok=True)

    regex = re.compile(scen + "_[0-9][0-9][0-9]_" + str(syear) + "_" + str(eyear) + ".*")
    infnames = [baseoutfolder + scen + "/" + i for i in  os.listdir(baseoutfolder + scen) if regex.match(i)]

    # items = [{"filename" : filename, "dset" : xr.open_dataset(filename), "member" : re.sub(scen + "_([0-9][0-9][0-9])_.*","\g<1>", os.path.basename(filename))} for filename in infnames]
    # Load data into a list of dicts with the data, the filename and the member
    print("Loading inputs...")
    items = [{"filename" : filename, "dset" : drop_unused_vars(xr.open_dataset(filename)), "member" : re.sub(scen + "_([0-9][0-9][0-9])_.*","\g<1>", os.path.basename(filename))} for filename in infnames]

    # Expand a "member" dimension in each DataSet and return a simple list of them
    print("Expanding dimensions...")
    expitems = [item["dset"].expand_dims("member") for item in items]

    #%%
    # Combine the DataSets on the "member" dimension and take the ensemble mean 
    # print("Combining and taking ensemble mean...")
    # dsmean = xr.combine_nested(expitems, concat_dim="member", combine_attrs= "override").mean(dim = "member", keep_attrs = True)
    
    print("Combining ...")
    bigds = xr.combine_nested(expitems, concat_dim="member", combine_attrs= "override")

    # Separate Datasets for means and variances
    regexvar = re.compile(".*_var$")
    dsmeans = bigds[[i for i in bigds.data_vars if not regexvar.match(i)]]
    # Rename variance variable names to match those of the means for pooling
    dsvars = bigds[[i for i in bigds.data_vars if regexvar.match(i)]]
    dsvars = dsvars.rename({i:(re.sub("(.*)_var$","\g<1>",i)) for i in dsvars.data_vars})


    # Aggregate means by taking a simple mean
    dsmean = dsmeans.mean(dim = "member", keep_attrs = True)

    # Aggregate variances by applying pooled_
    dsvariance = xr.apply_ufunc(pool_variances_dask, dsmeans, dsvars, nyears, \
        dask = "parallelized", input_core_dims=[["member"], ["member"], [] ],
        keep_attrs=True)

    # Join variance variables into dsmean
    dsvariance = dsvariance.rename({i:(i + "_var") for i in dsvariance.data_vars})
    dsmean = dsmean.merge(dsvariance)

    #%%
    # With the climatological monthly ensemble means, we will now interpolate 
    # levels on each 4D variable
    print("Preparing for interpolation...")
    # List of variables to interpolate
    intvarnames = [var for var in dsmean.data_vars if dsmean[var].shape.__len__() == 4]

    # Extract numpy arrays of the relevant variables for Ngl.vinth2p
    # We'll need the name of time variable, could be "time" or "month" at this point
    # We need it because previous steps likely copied these variables over all times
    timvarname = [i for i in ["time", "month"] if i in list(dsmean.coords)][0]

    hyam = dsmean["hyam"][{timvarname : 0}] # Hybrid level coefficient A
    hybm = dsmean["hybm"][{timvarname : 0}] # Hybrid level coefficient B
    P0mb =  0.01*dsmean["P0"][{timvarname : 0}] # Reference pressure (scalar)

    psrf = dsmean["PS"] # Pressure at surface (time-dependent)

    lats = dsmean["lat"]
    lons = dsmean["lon"]
    times = dsmean[timvarname]

    # Define a coordinate variable for the output pressure levels
    # Set units to "hPa" here, but will override later when converting 
    lev = xr.DataArray(pnew, coords = [pnew], dims = "lev", attrs = {"units" : "hPa"})

    # DataSet that will contain all output
    dsout = xr.Dataset()

    #%%
    # 4D variables loop
    # varname = intvarnames[1]
    for varname in intvarnames:
        print("Interpolating " + varname)
        var = dsmean[varname]

        # Actual interpolation
        varnew = Ngl.vinth2p(var,hyam,hybm,pnew,psrf,intyp,P0mb,1,kxtrp)

        # Put the interpolated variable in a DataSet with it's metadata
        dsnew = xr.Dataset({varname: ((timvarname,'lev','lat','lon'), varnew[:,:,:,:])},
                        {'lon':  lons, 'lat':  lats, timvarname: times, 'lev' : lev})

        # Convert vertical coordinate from hPa to Pa
        dsnew["lev"] = dsnew["lev"]*100.0
        dsnew["lev"].attrs['units'] = 'Pa'

        # Copy metadata from the original variable
        dsnew[varname].name = var.name
        dsnew[varname].attrs = var.attrs

        # Merge this DataSet with the output one, that starts out empty
        dsout = dsout.merge(dsnew)

    # Merge in the 2D variables
    dsout = dsout.merge(dsmean[list(set([i for i in list(dsmean.data_vars) if dsmean[i].shape.__len__() == 3]) - \
        set(dsout.data_vars))])

    # Copy attributes from the original dataset and add some
    dsout.attrs = dsmean.attrs
    dsout.attrs['STEP2_comment'] = 'Vertical levels interpolated via Ngl.vinth2p using method (intyp) ' + str(intyp) + " and extrapolation " + str(kxtrp) + " on " + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    dsout.attrs['STEP2_original_levels'] = str(dsmean.lev)

    #%%
    # Save output file
    outfname = outfolder + scen + "_ensmean_pres_"+ str(syear) + "_" + str(eyear) + ".nc"
    dsout.to_netcdf(outfname)

    # Cleanup
    del(dsout)
    del(dsmean)
    del(dsvariance)
    del(dsmeans)
    del(dsvars)
    del(bigds)
    del(expitems)
    del(items)


# #%%
# bigv = np.random.normal(4,1,40)
# np.var(bigv)
# n = 10
# subv = [bigv[i:i+n] for i in range(0,len(bigv),n)]
# means = [np.mean(i) for i in subv]
# variances = [np.var(i, ddof = 1) for i in subv]

# @jit(nopython = True)
# def pool_pair_means(a,b,rn):
#     out = (rn*a + b)/(rn+1)
#     return(out)

# @jit(nopython = True)
# def pool_means(v):
#     i = 0
#     s = v[0]
#     while i < (len(v)-1):
#         rn = i + 1
#         s = pool_pair_means(s,v[i+1],rn)
#         i = i + 1
#     return(s)

# @jit(nopython = True)
# def pool_pair_variances(v1,v2,m1,m2,n1,n2):
#     vp = (1/(n1+n2-1)) * ( (n1-1)*v1 + (n2-1)*v2 + ((n1*n2)/(n1+n2)) * (m1-m2)**2 )
#     return(vp)

# @jit(nopython = True)
# def pool_variances(means, variances,n):
#     i = 0
#     vp = variances[0]
#     mp = means[0]
#     while i < (len(variances)-1):
#         rn = i + 1
#         vp = pool_pair_variances(vp, variances[i+1], mp, means[i+1], n*rn, n)
#         mp = pool_pair_means(mp,means[i+1],rn)
#         i = i + 1
#     return(vp)


# pool_variances(means,variances,n)
# np.var(bigv, ddof = 1)

# pool_pair_variances(variances[0],variances[1],means[0],means[1],len(subv[0]),len(subv[1]))
# np.var(np.concatenate([subv[0],subv[1]]), ddof =1 )

# pool_means(means)
# np.mean(bigv)
# subsoma(1,4)

#%% Old code
# #%% Folders (members) loop
#     # infolder = infolders[0]
#     for infolder in infolders:

#         memberstr = (os.path.split(os.path.split(os.path.split(infolder)[0])[0])[1]).split(sep = "_")[-1]

#         # filenames = glob.glob(infolder + '*{2030..2050}*')
#         filenames = [glob.glob(infolder + "*." + mon + ".*")  for mon in [str(year) + "-" + i for year in years for i in [str(j).zfill(2) for j in range(1,13)]]]
#         filenames = [i for j in filenames for i in j] # Flatten it

#         #%% Open a folder's (member) dataset
#         inds = xr.open_mfdataset(filenames)
#         tmeands = inds.groupby("time.month").mean()

#         #%% Save output. Since we're lazy loading, this is where most of the processing happens
#         outfname = outfolder + scen + "_" + memberstr + "_" + str(syear) + "_" + str(eyear) + ".nc"
#         print(outfname + " | " + datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
#         tmeands.to_netcdf(outfname)


# #%%

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

# lats = ds["lat"]
# lons = ds["lon"]
# times = ds['time']

# var    = ds[varname]

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
# %%
