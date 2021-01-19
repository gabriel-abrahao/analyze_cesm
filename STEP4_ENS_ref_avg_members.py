# STEP4 of preprocessing raw 3D files
# Averages (and takes the variance) of all ensemble members


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
# %matplotlib inline

from pooled_stats import *

#%%
# filename = "/media/gabriel/hd2_6tb/ccsm4/test/rcp0.0_veg_005.cam2.h0.1950-01.nc"

scen = "historical_0.0_heg"
membercodes = ["r1i1p1","r1i2p1","r2i1p1","r6i1p1"]

baseinfolder = "refdata/historical/allens/"
baseoutfolder = "refdata/historical/"

syear = 1990
eyear = 2005

# Number of years, we may need it to calculate degrees of freedom if the nobs attribute
# in the inputs is not set properly. Will be overriden if it is
nyears = eyear - syear + 1

# Try to keep the format from the processed RCP simulations
outfname = baseoutfolder + "historical_0.0_heg_ensmean_pres_" + str(syear) + "_" + str(eyear) + ".nc"

#%%
# =================== FUNCTION DEFINITIONS
# Add a suffix to all variables in a Dataset
def add_suf(ds,suf):
    return(ds.rename({i:(i + suf) for i in ds.data_vars}))

# Removes a suffix from all variables in a Dataset
def rem_suf(ds,suf):
    return(ds.rename({i:(re.sub("(.*)"+suf+"$","\g<1>",i)) for i in ds.data_vars}))

# Splits a dataset that has means and variances coded by name. 
# e.g. the mean of variable X is in variable X and it's variance is in X_val
def split_dataset_variances(dsboth):
    regexvar = re.compile(".*_var$")
    dsmeans = dsboth[[i for i in dsboth.data_vars if not regexvar.match(i)]]
    dsvariances = dsboth[[i for i in dsboth.data_vars if regexvar.match(i)]]
    dsvariances = dsvariances.rename({i:(re.sub("(.*)_var$","\g<1>",i)) for i in dsvariances.data_vars})

    # Using slice like this here ensures lev is kept as a singleton dimension
    # dsmeans = dsmeans[selvars].sel(lev=slice(uselev,uselev))
    # dsvariances = dsvariances[selvars].sel(lev=slice(uselev,uselev))
    return((dsmeans, dsvariances))

#%%
# =================== BEGIN SCRIPT
# Get all filenames
allfnames = glob.glob(baseinfolder + scen + "*")

#%%
# Create output folder
os.makedirs(baseoutfolder, exist_ok=True)

#%%
# Open all files and concatenate them over a 'member' dimension
# We can just concatenate over members without worrying about member names    
items = [xr.open_dataset(f).expand_dims("member") for f in allfnames]

# WARNING: We're getting some rounoff error in dimensions, should tweak tolerance if using for other application
# This step has the side effect of loading everything to memory,
# so we need another solution if lazy loading is required
items = [item.reindex_like(items[0], method = "nearest", tolerance = 0.001) for item in items]

# Now concatenate
inds = xr.combine_nested(items,  concat_dim="member", combine_attrs= "override")

#%%
# Split means and variances in separate datasets

(dsmeans, dsvariances) = split_dataset_variances(inds)

# Get nyears from attributes if there are any
# Variable attributes take precedence over global ones
if "nobs" in inds[list(inds.data_vars)[0]].attrs.keys():
    nyears = inds[list(inds.data_vars)[0]].attrs["nobs"]
elif "nobs" in inds.attrs.keys():
    nyears = inds.attrs["nobs"]

# Aggregate means by taking a simple mean
dsmean = dsmeans.mean(dim = "member", keep_attrs = True)

# Aggregate variances by applying function from pooled_stats.py
dsvariance = xr.apply_ufunc(pool_variances_dask, dsmeans, dsvariances, nyears, \
    dask = "parallelized", input_core_dims=[["member"], ["member"], [] ],
    keep_attrs=True)

# Join variance variables into dsmean
dsvariance = dsvariance.rename({i:(i + "_var") for i in dsvariance.data_vars})
dsmean = dsmean.merge(dsvariance)

#%%
# Save to output file
# This may appear to be the bottleneck, but the pooling of variances
# uses dask so the actual computation happens here.
dsmean.to_netcdf(outfname)

# #%%
# # Means
# tmeands = inds.mean(dim = ["time", "member"], keep_attrs = True)

# # Variances
# tvards = inds.var(dim = ["time", "member"], keep_attrs = True, ddof = 1)
# tvards = tvards.rename({i:(i + "_var") for i in tvards.data_vars})

# toutds = xr.merge([tmeands,tvards])

# toutds.attrs['nobs_var'] = inds.dims['member']*inds.dims['time']
