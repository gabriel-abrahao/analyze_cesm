# Creates daily agriculturally relevant variable files
# based on model output on the CMIP standard format, not base CESM
#
# This and daily_ag_from_cesm.py will create output that is 
# compatible with each other
# 
# WARNING: CCSM4 (CESM) on CMIP5 did not output daily humidity, so we need to
# interpolate QREFHT from monthly files.
# To keep it consistent, we must average and reinterpolate daily QREFHT
# when using raw CESM output where they are available (SEG, WEG and VEG scenarios).
# %%
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cftime
import tqdm
import os
import glob
import re
import copy
import datetime
import numba
from numba import jit,prange

#%%
baseoutfolder       =   "output_ag/ag_daily/"
icleanout           =   True

baseinfolder        =   "refdata/ccsm4/"

dayfoldersuf        =   "daily"
monfoldersuf        =   "monthly"

scenarios           =   ["rcp2.6_cmp","rcp8.5_cmp","historical"]

# File prefixes of each variable
varprefs            =   ["pr","psl","tas","tasmax","tasmin"]

monvarpref          =   "huss"

# Full period, including the used range of all experiments
fullsyear           =   1990
fulleyear           =   2050

# Convert CMIP ensemble member names to loosely match CESM numbers.
# These should ensure no repeating names for the same scenario
# Be aware that they are not continuation of the same runs
ensmemdict          =   {
    "r1i1p1"   :   "005",
    "r2i1p1"   :   "006",
    "r3i1p1"   :   "008",
    "r4i1p1"   :   "009",
    "r1i2p1"   :   "008",
    "r6i1p1"   :   "009",
}


#%% Function definitions
# =================================================================================================
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

# Drop bounds and extract DataArray with only one actual variable
# There should be only one variable per prefix. Here we drop "_bnds" variables
# and get a DataArray with only that variable
def drop_bounds(inds):
    varcandidates = [i for i in inds.data_vars if not re.match(".*_bnds$",i)]
    if len(varcandidates) >1:
        print("WARNING: More than one variable in files!")
    varname = varcandidates[0]
    dayinar = inds[varname]
    return(dayinar)
    

# Writes or appends DataArray to a NetCDF file
# If the file doesn't exist yet (i.e. the first run),
# use write mode to create it, else use append mode
def nc_write_append_dataarray(da, outfname):
    if not os.path.exists(outfname):
        da.to_netcdf(outfname, mode = 'w')
    else:
        da.to_netcdf(outfname, mode = 'a')

# Adds a number of days to a cftime object, tolerating errors
# Errors shouldn't be larger than a day for cftime.DatetimeNoLeap
def add_days(indate,ndays):
    dumdate = datetime.date(indate.year,indate.month,indate.day)
    dumdate = dumdate + datetime.timedelta(days = ndays)
    outdate = cftime.DatetimeNoLeap(dumdate.year,dumdate.month,dumdate.day)
    return(outdate)

# Pads a DataArray by repeating the first and the last timesteps
# 31 days before and after
def pad_months_repeat(inar):
    mindate = inar.time.min().item()
    maxdate = inar.time.max().item()

    firstmonth = inar.sel(time = mindate)
    firstmonth["time"] = add_days(firstmonth.time.item(),-31)

    lastmonth = inar.sel(time = maxdate)
    lastmonth["time"] = add_days(lastmonth.time.item(),31)

    outar = xr.concat([firstmonth,inar,lastmonth], dim = "time")
    return(outar)
# =================================================================================================
#%%
# Create time slice TODO: We could support more calendars here if we put this inside the loops
fullslice = slice(cftime.DatetimeNoLeap(fullsyear,1,1),cftime.DatetimeNoLeap(fulleyear+1,1,1))

# scen = scenarios[2]
# scen = scenarios[0]
# == SCENARIOS LOOP
for scen in scenarios:
    sceninfolder = baseinfolder + "/" + scen + "/"
    print(sceninfolder)

    # ensmemin = "r1i2p1"
    # == ENSEMBLE MEMBERS LOOP
    for ensmemin in ensmemdict.keys():
        ensmemout = ensmemdict[ensmemin]
        print(ensmemin + " | " + ensmemout)

        ensmemfolder = sceninfolder + ensmemin + "/"
        #     if os.path.exists(ensmemfolder):
        #         print(ensmemfolder)

        outensmemfolder = baseoutfolder + "/" + scen + "_" + ensmemout + "/"

        if not os.path.exists(ensmemfolder):
            print("No member " + ensmemin + " for scenario " + scen + ", skipping...")
            continue

        # Create the output folder
        os.makedirs(outensmemfolder, exist_ok=True)

        # Clean the output folder if requested to keep it consistent
        if icleanout:
            for i in os.listdir(outensmemfolder):
                os.remove(outensmemfolder + i)

        # First operate on the daily variables
        dayinfolder = ensmemfolder + "/" + dayfoldersuf + "/"
        
        print("Copying daily...")
        # == VARIABLES LOOP (DAILY)
        # varpref = varprefs[0]
        for varpref in varprefs:
            dayinfnames = glob.glob(dayinfolder + varpref + "_*")

            # Read all files in a dataset
            dayinds = xr.open_mfdataset(dayinfnames)

            # Cut only desired years
            dayinds = dayinds.sel(time = fullslice)

            # Drop bounds
            dayinar = drop_bounds(dayinds)

            # Rename variable
            dayinar.name = get_original_name(dayinar)

            # == YEARLY LOOP (DAILY)
            for year in tqdm.tqdm(range(fullsyear,fulleyear+1), desc=varpref):

                yearslice = slice(cftime.DatetimeNoLeap(year,1,1),cftime.DatetimeNoLeap(year+1,1,1))

                # Yearly array (may be empty, time length zero)
                ydayar = dayinar.sel(time = yearslice)

                # Reproduce a CESM history file name 
                yfname = outensmemfolder + scen + "_" + ensmemout + ".cam2.h1." + str(year) + "-01-01-00000.nc"

                if len(ydayar.time) != 0:
                    # Writes/appends to output. 
                    nc_write_append_dataarray(ydayar,yfname)

        # THIS SHOULD BE RIGHT OUTSIDE THE VARIABLES LOOP
        # Now we interpolate the monthly variable
        print("Iterpolating monthly...")
        # We should have a leftover in dayinarr to get its time
        daytime = dayinar.time

        moninfolder = ensmemfolder + "/" + monfoldersuf + "/"

        moninfnames = glob.glob(moninfolder + monvarpref + "_*")

        # Read all files in a dataset
        moninds = xr.open_mfdataset(moninfnames)

        # Cut only desired years
        moninds = moninds.sel(time = fullslice)

        # Obtain DataArray, transposing to match daily dimension orders
        moninar = drop_bounds(moninds).transpose(dayinds)

        # Rename variable
        moninar.name = get_original_name(moninar)

        # Repeat first and last months for extrapolation
        moninar = pad_months_repeat(moninar)

        # == YEARLY LOOP (DAILY)
        for year in tqdm.tqdm(range(fullsyear,fulleyear+1)):

            # A time vector with the days of that year
            # Selecting from daytime ensures metadata are the same
            # and allows us to test for its size
            ydaytime = daytime.sel(time = slice(cftime.DatetimeNoLeap(year,1,1),cftime.DatetimeNoLeap(year+1,1,1)))

            if len(ydaytime) != 0:
                # Do the actual interpolation
                outintmonar = moninar.interp(time = ydaytime)

                # Yearly file name
                yfname = outensmemfolder + scen + "_" + ensmemout + ".cam2.h1." + str(year) + "-01-01-00000.nc"

                # Writes/appends to output. 
                nc_write_append_dataarray(outintmonar,yfname)
