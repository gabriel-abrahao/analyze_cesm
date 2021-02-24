# Creates a calendar with the same dates in all pixels
# %%
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cftime
import tqdm
import os
import re
import copy
import datetime
import numba
from numba import jit,prange

#%%

# Use reference calendar files to ensure consistency
refcalfolder    =   "../input/calendars/ming595_mean/"

basecalfolder   =   "../input/calendars/"

calname         =   "Oct15-Mar1_Jan15-Jun15"

crops       =   ["single", "maize"]

# Here we set the calendar as well. Preferably use a pair of nonleap years
# We could have just used cftime.DatetimeNoLeap, but those don't support any operations
# sdate           =   cftime.DatetimeNoLeap(2001  ,10 ,1  )
sdates            =   {}
edates            =   {}

sdates["single"]            =   datetime.datetime(2005  ,10 ,15  )
edates["single"]            =   datetime.datetime(2006  ,3 ,1  )

sdates["maize"]            =   datetime.datetime(2005  ,1 ,15  )
edates["maize"]            =   datetime.datetime(2006  ,6 ,15  )

outfolder           =   basecalfolder + "/" + calname + "/"

#%%
# Create output folder
os.makedirs(outfolder, exist_ok=True)

# Get DOYs. Remember that calendars for use in compute_gs_vars are just DOYs, daysyear is defined there
sdoys   =   {i:float(sdates[i].timetuple().tm_yday) for i in sdates.keys()}
edoys   =   {i:float(edates[i].timetuple().tm_yday) for i in edates.keys()}

# crop = crops[0]
for crop in crops:
    reffname            =   refcalfolder + "/" + crop + ".crop.calendar.fill.nc"
    outfname            =   outfolder + "/" + crop + ".crop.calendar.fill.nc"

    # Read base calendar
    refcalds = xr.open_dataset(reffname)

    outcalds = copy.deepcopy(refcalds)

    # outcalds["plant.start"] = xr.broadcast(outcalds["plant.start"],sdoys[crop])

    outcalds["plant.start"] = xr.where(np.isfinite(outcalds["plant.start"]), sdoys[crop], np.nan)
    outcalds["harvest.end"] = xr.where(np.isfinite(outcalds["harvest.end"]), edoys[crop], np.nan)

    # Write output
    outcalds.to_netcdf(outfname)