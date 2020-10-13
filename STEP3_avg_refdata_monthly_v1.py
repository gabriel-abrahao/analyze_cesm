# STEP3 of preprocessing raw 3D files
# Preprocess reference period data into a single file
# Change units and variable names to match our simulation's output
# Levels should already be the same after interpolating in STEP2
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

#%%
# filename = "/media/gabriel/hd2_6tb/ccsm4/test/rcp0.0_veg_005.cam2.h0.1950-01.nc"

base2dmoninfolder = "refdata/ccsm4/2d/r1l1p1/monthly/"
base2ddayinfolder = "refdata/ccsm4/2d/r1l1p1/daily/"
base3dmoninfolder = "refdata/ccsm4/3d/r1l1p1/"
baseoutfolder = "refdata/historical/"

syear = 1990
eyear = 2005

# Try to keep the format from the processed RCP simulations
outfname = baseoutfolder + "historical_0.0_heg_ensmean_pres_" + str(syear) + "_" + str(eyear) + ".nc"

#%% Function definitions

# Gets the original (CCSM4) name of the variable from
# the original_name attribute, correcting for specific cases
# where the variable was derived from more than one CCSM4 variable
def get_original_name(da):
    origstring = da.attrs["original_name"]
    if origstring == "PRECC,PRECL":
        newname = "PRECT"
    elif origstring == "Z3,PS":
        newname = "Z3"
    else:
        newname = origstring
    return(newname)

# Converts specified units
# We're not enforcing unit names here, just converting select ones
# For example, Pa/s and Pa s-1 won't be made the same
def convert_units(da):
    if da.attrs['units'] == "kg m-2 s-1":
        # Convert from kg m-2 s-1 to m/s
        da.values = da.values/1000
        da.attrs['units'] = "m/s"
    else:
        da = da
    return(da)
#%%
# Get all filenames
allfnames = [glob.glob(base2ddayinfolder + "*.nc"), glob.glob(base2dmoninfolder + "*.nc"), glob.glob(base3dmoninfolder + "*.nc")]
allfnames = [j for i in allfnames for j in i]

#%%
# Create output folder
os.makedirs(baseoutfolder, exist_ok=True)
# Remove output file if it exists, since we'll be appending to it
if os.path.exists(outfname):
    os.remove(outfname)

# %%
# fname = allfnames[2]
for fname in allfnames:
    print(fname)
    # Get the relevant variable name from the prefix in fname
    invarname = os.path.basename(fname).split("_")[0]

    # Open just the DataArray from the prefix, slicing the time
    dain = xr.open_dataset(fname)[invarname].sel(time = slice(str(syear),str(eyear)))

    # Climatological monthly average
    daout = dain.groupby("time.month").mean(keep_attrs = True)

    # Changes the name to the original (CCSM4) one
    daout.name = get_original_name(dain)

    # Converts some units
    daout = convert_units(daout)

    # Writes to output. 
    # If the file doesn't exist yet (i.e. the first run),
    # use write mode to create it, else use append mode
    if not os.path.exists(outfname):
        daout.to_netcdf(outfname, mode = 'w')
    else:
        daout.to_netcdf(outfname, mode = 'a')
