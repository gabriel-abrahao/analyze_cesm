# Creates daily agriculturally relevant variable files based on CESM files
# This basically copies some daily variables and degrades QREFHT as described below
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
icleanout           =   True    # Cleans each ensemble member's output folder before starting to process it

baseinfolder        =   "input/rcp_daily"

scenarios           =   ["rcp2.6_seg",  "rcp2.6_weg", "rcp8.5_seg",  "rcp8.5_weg"]

# Names of each variable
varnames            =   ["PRECT","PS","TREFHT","TREFHTMX","TREFHTMN"]

monvarpref          =   "QREFHT"

allvarnames = varnames.copy()
allvarnames.append(monvarpref)

# Full period, including the used range of all experiments
fullsyear           =   1990
fulleyear           =   2050

# Convert ensemble member names to something. Shouldn't be necessary at this time
# when using CESM, but keeping the option with a mirrored dict for now.
# These should ensure no repeating names for the same scenario
# Be aware that they are not continuation of the same runs
ensmemdict          =   {
    "005"   :   "005",
    "006"   :   "006",
    "008"   :   "008",
    "009"   :   "009"
}

# Domain
minlat = -61
maxlat = 15
minlon = 273
maxlon = 345

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

    outar = xr.concat([firstmonth,inar,lastmonth], dim = "time").transpose(*inar.dims)
    return(outar)

# =================================================================================================
#%%
# Create time slice TODO: We could support more calendars here if we put this inside the loops
fullslice = slice(cftime.DatetimeNoLeap(fullsyear,1,1),cftime.DatetimeNoLeap(fulleyear+1,1,1))

# scen = scenarios[0]
# == SCENARIOS LOOP
for scen in scenarios:
    sceninfolder = baseinfolder + "/" + scen + "_"
    print(sceninfolder)

    # ensmemin = "005"
    # == ENSEMBLE MEMBERS LOOP
    for ensmemin in ensmemdict.keys():
        ensmemout = ensmemdict[ensmemin]
        print(ensmemin + " | " + ensmemout)

        ensmemfolder = sceninfolder + ensmemin + "/"
        #     if os.path.exists(ensmemfolder):
        #         print(ensmemfolder)

        outensmemfolder = baseoutfolder + "/" + scen + "_" + ensmemout + "/"
        print(outensmemfolder)

        if not os.path.exists(ensmemfolder):
            print("No member " + ensmemin + " for scenario " + scen + ", skipping...")
            continue


        # Create the output folder
        os.makedirs(outensmemfolder, exist_ok=True)

        # Clean the output folder if requested to keep it consistent
        if icleanout:
            for i in os.listdir(outensmemfolder):
                os.remove(outensmemfolder + i)

#%%
        # First operate on the daily variables
        dayinfolder = ensmemfolder + "/run/"

        dayinfnames = glob.glob(dayinfolder + "/" + scen + "_" + ensmemin + ".*.nc")

        print("Copying daily...")
        # == VARIABLES LOOP (DAILY)
        # Read all files in a dataset and cut only desired years and variables
        bigdayinds = xr.open_mfdataset(
            dayinfnames, data_vars = allvarnames, parallel = True
            )[allvarnames].sel(time = fullslice, lat = slice(minlat,maxlat), lon = slice(minlon,maxlon))
        
        # varpref = varnames[0]
        # varpref = "TREFHTMN"
        for varpref in varnames:

            # Extract the variable as a DataArray
            dayinar = bigdayinds[varpref]

            # year = 2005
            # == YEARLY LOOP (DAILY)
            for year in tqdm.tqdm(range(fullsyear,fulleyear+1), desc=varpref):

                # Apparently we need 31/12/year for CESM and 01/01/year+1 for CMIP
                yearslice = slice(cftime.DatetimeNoLeap(year,1,1), \
                    add_days(cftime.DatetimeNoLeap(year+1,1,1,0,0,0,0),-1))

                # Yearly array (may be empty, time length zero)
                ydayar = dayinar.sel(time = yearslice)

                # Check if the year has exactly 365 days, or skip it if zero
                # Some issues may arise on yearslice with the last day that
                # can lead to 364 or 366 days.
                if len(ydayar.time) == 0:
                    continue
                elif len(ydayar.time) != 365:
                    raise NameError("Year " + str(year) + " has " + str(len(ydayar.time)) + "days, check yearslice")


                # Reproduce a CESM history file name 
                yfname = outensmemfolder + scen + "_" + ensmemout + ".cam2.h1." + str(year) + "-01-01-00000.nc"

                # Writes/appends to output. 
                nc_write_append_dataarray(ydayar,yfname)

        # THIS SHOULD BE RIGHT OUTSIDE THE VARIABLES LOOP
        # Now we interpolate the monthly variable
        # Since we have daily files, we will average first and then 
        # interpolate normally to ensure consistency
        print("Iterpolating monthly...")
        # We should have a leftover in dayinarr to get its time
        daytime = dayinar.time

        # Get the original daily variable
        dayinar = bigdayinds[monvarpref]

        # Resample. Parameters shift labels to be centered on the 15th of each month
        moninar = dayinar.resample(time = "1M", label="left", loffset="15D").mean()

        # Repeat first and last months for extrapolation
        moninar = pad_months_repeat(moninar)

        # == YEARLY LOOP (DAILY)
        for year in tqdm.tqdm(range(fullsyear,fulleyear+1)):

            # Apparently we need 31/12/year for CESM and 01/01/year+1 for CMIP
            yearslice = slice(cftime.DatetimeNoLeap(year,1,1), \
            add_days(cftime.DatetimeNoLeap(year+1,1,1,0,0,0,0),-1))            

            # A time vector with the days of that year
            # Selecting from daytime ensures metadata are the same
            # and allows us to test for its size
            # ydaytime = daytime.sel(time = slice(cftime.DatetimeNoLeap(year,1,1),cftime.DatetimeNoLeap(year+1,1,1)))
            ydaytime = daytime.sel(time = yearslice)

            if len(ydaytime) != 0:
                # Do the actual interpolation
                outintmonar = moninar.interp(time = ydaytime)

                # Yearly file name
                yfname = outensmemfolder + scen + "_" + ensmemout + ".cam2.h1." + str(year) + "-01-01-00000.nc"

                # Writes/appends to output. 
                nc_write_append_dataarray(outintmonar,yfname)

