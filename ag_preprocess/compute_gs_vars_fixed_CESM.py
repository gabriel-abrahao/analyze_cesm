# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
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

# %% [markdown]
# ### Setup

# %%
# infolder = "../external/data/xavier/daily_all/"
baseinfolder = "../output_ag/preprocessed/"
baseoutfolder = "../output_ag/dd_calc/"

# models      =   ["BCC-CSM2-MR","CanESM5","EC-Earth3-Veg","GFDL-CM4","MRI-ESM2-0"]
# scens       =   ["historical", "ssp245", "ssp585"]
scens       =   next(os.walk(baseinfolder))[1]

crops = ["single","maize"]

calname = "ming595_mean"

temppref = "temp.daily."
tmaxpref = "tmax.daily."
tminpref = "tmin.daily."
precpref = "prec.daily."
vpdpref = "vpd.daily."

# calfolder = "fixed_calendars/"+calname+"/"
calfolder = "../input/calendars/"+calname+"/"
calsuf = ".crop.calendar.fill.nc"
planvar = "plant.start"
harvvar = "harvest.end"

# Parameters for evaluating the temperature distribution
# Tlo = -5.0
Tlo = 10.0
# Thi = 45.0
Thi = 32.0
Tint = 1.0

# Dirty override of the scenario counter, can be used to restart runs
ioverscen = False
# ioverscen = True
overscenindex = 2
if ioverscen:
    print("====================================================\n===== WARNING: Skipping to scenario index " + str(overscenindex) + " ========\n====================================================")


# %% [markdown]
# ### Defining functions

# %%
# Opens and concatenates a harvest year with the equivalent planting year
def concat_clim(infolder,climpref,hyear):
    pyear = hyear - 1

    climharr = xr.open_dataarray(infolder+climpref+str(hyear)+".nc")
    climparr = xr.open_dataarray(infolder+climpref+str(pyear)+".nc")

    climarr = xr.concat([climparr,climharr], dim = "time")
    return climarr

# Shifts times of day to 0 hour of that day inplace, from a DataArray/Dataset
def shift_hour_midnight(inar):
    inar["time"] = inar["time"] = [i.replace(hour=0, minute=0,second=0) for i in inar.time.values.tolist()]
    return(None)

# Opens two years (planting and harvest) given a folder, a prefix and a harvest year
def read_cfnoleap_year(infolder,climpref,hyear):

    # timerange = xr.cftime_range(cftime.DatetimeNoLeap(hyear-1, 1, 1, 0, 0, 0, 0),cftime.DatetimeNoLeap(hyear, 12, 31, 0, 0, 0, 0))
    timerange = xr.cftime_range(cftime.DatetimeNoLeap(hyear-1, 1, 1),cftime.DatetimeNoLeap(hyear, 12, 31))

    climarr = xr.open_dataarray(infolder + "/" + climpref + "allyears.nc")
    # Shift hour of day to zero, this is what timerange expects
    shift_hour_midnight(climarr)
    climarr = climarr.sel(time = timerange)
    return climarr


# %%
# Calculates t distribution for a single day
@jit(nopython=True)
def calc_dist_day(Tmin,Tmax,Tl1s,Tint):
    res = 0.005 #Resolution (dt, in days) on which to evaluate the T sine curve

#     Tl1s = np.arange(Tlo,Thi,Tint)
    nT = Tl1s.shape[0]
    exps = np.zeros_like(Tl1s)

    t = np.arange(0,1,res)
    nt = t.shape[0]

    Tamp = (Tmax-Tmin)/2.0
    Tmed = (Tmax+Tmin)/2.0

    T = Tmed + Tamp*np.sin(t*(2.0*np.pi/1))

    for tcount in range(nT):
        Tl1 = Tl1s[tcount]
        Tl2 = Tl1 + Tint
        exps[tcount] = np.sum(np.invert(np.isnan(
            np.where((T>=Tl1) & (T<=Tl2) ,T,np.nan)
        )))/nt
    return exps


@jit(nopython=True)
def calc_dist_point(tmaxvec,tminvec,Tlos,Tint):
    allexps = np.zeros_like(Tlos)
    for day in range(tminvec.shape[0]):
        allexps = allexps + calc_dist_day(tminvec[day],tmaxvec[day],Tlos,Tint)
    return allexps


# %%
# Calculates GDD for a single day
@jit(nopython=True)
def calc_gdd_day(Tmin,Tmax,Tl1s):
    res = 0.005 #Resolution (dt, in days) on which to evaluate the T sine curve

    #     Tl1s = np.arange(Tlo,Thi,Tint)
    nT = Tl1s.shape[0]
    gdds = np.zeros_like(Tl1s)

    t = np.arange(0,1,res)
    nt = t.shape[0]

    Tamp = (Tmax-Tmin)/2.0
    Tmed = (Tmax+Tmin)/2.0

    T = Tmed + Tamp*np.sin(t*(2.0*np.pi/1))

    for tcount in range(nT):
        Tl1 = Tl1s[tcount]
        gdds[tcount] = np.sum(np.invert(np.isnan(
            np.where((T>=Tl1),T,np.nan)
        ))*(T-Tl1))/nt

    return gdds

@jit(nopython=True)
def calc_gdd_point(tmaxvec,tminvec,Tlos):
    allgdds = np.zeros_like(Tlos)
    for day in range(tminvec.shape[0]):
        allgdds = allgdds + calc_gdd_day(tminvec[day],tmaxvec[day],Tlos)
    return allgdds


# %%
# # Calculates VPD for a vector
# # Assuming sea level pressure everywhere
# @jit(nopython=True)
# def calc_vpd(q,tmin,tmax):
#     e = 1013.25 *(q/(1-q))/(0.622 + (q/(1-q)))

#     esmin = 6.1078*np.exp((17.3*tmin)/(237.3+tmin))
#     esmax = 6.1078*np.exp((17.3*tmax)/(237.3+tmax))
#     es = (esmin+esmax)/2.0

#     vpd = es - e
#     vpd = np.where(vpd < 0,0,vpd)


# %%
# Calculates everything for the growing season given numpy arrays. 
# Loops throught the points and calculates both distribution and regular gs means
# Compiles with Numba parallel
@jit(nopython=True, parallel = True)
# @jit(nopython=True)
def calc_all(planmat,harvmat,tempmat,tmaxmat,tminmat,precmat,
             vpdmat,
             tempmeanmat,tmaxmeanmat,tminmeanmat,precmeanmat,
             vpdmeanmat,
             trngmeanmat,ndaymat,
             tempdistmat,tempgddsmat):
    
    def calc_vpd(q,tmin,tmax):
        e = 1013.25 *(q/(1-q))/(0.622 + (q/(1-q)))

        esmin = 6.1078*np.exp((17.3*tmin)/(237.3+tmin))
        esmax = 6.1078*np.exp((17.3*tmax)/(237.3+tmax))
        es = (esmin+esmax)/2.0

        vpd = es - e
        vpd = np.where(vpd < 0,0,vpd)
        return vpd
    
    for lati in prange(tempmat.shape[1]):
        for lonj in range(tempmat.shape[2]):
            if (np.isnan(planmat[lati,lonj])) or (np.isnan(tempmat[0,lati,lonj])):
                continue
            plan = int(planmat[lati,lonj])
            harv = int(harvmat[lati,lonj])

            tempvec = tempmat[plan:harv,lati,lonj]
            tmaxvec = tmaxmat[plan:harv,lati,lonj]
            tminvec = tminmat[plan:harv,lati,lonj]
            precvec = precmat[plan:harv,lati,lonj]
            vpdvec = vpdmat[plan:harv,lati,lonj]
            
#             vpdvec = calc_vpd(sphmvec,tminvec,tmaxvec)

            tempmeanmat[lati,lonj] = np.nanmean(tempvec)
            tmaxmeanmat[lati,lonj] = np.nanmean(tmaxvec)
            tminmeanmat[lati,lonj] = np.nanmean(tminvec)
            precmeanmat[lati,lonj] = np.nanmean(precvec)
            
            vpdmeanmat[lati,lonj] = np.nanmean(vpdvec)
            
            trngmeanmat[lati,lonj] = np.nanmean(tmaxvec - tminvec)
            ndaymat[lati,lonj] = np.int64(harv-plan)
            # tempdistmat[:,lati,lonj] = calc_dist_point(tmaxvec,tminvec,Tlos,Tint)
            tempgddsmat[:,lati,lonj] = calc_gdd_point(tmaxvec,tminvec,Tlos)

# %% [markdown]
# ================================================================================================================
# ### Begin main script

# %%
# iscen = 2
for iscen in range(len(scens)):
    # Scenario index override
    if ioverscen:
        if iscen < iscen:
            continue

    scen = scens[iscen]
    print(scen)

    infolder = baseinfolder + scen + "/"
    outfolder = baseoutfolder + "/" + calname + "/" + scen + "/"

    # Get the time variable of a file to infer years
    dumtime = xr.open_dataarray(infolder + "/" + temppref + "allyears.nc").time
    syear = dumtime.min().item().year
    eyear = dumtime.max().item().year

    # The basis of calculation will be harvest years
    hyears = list(range(syear+1,eyear+1)) # Remember that range() doesn`t include the last element
    # hyears = list(range(syear,eyear+1))


    # Create output folder
    if not os.path.exists(outfolder): os.makedirs(outfolder, exist_ok=True)

    #FIXME: Loop crops here
    # crop = crops[0]
    for crop in crops:
        print(crop)

        # Open calendar and convert it to two-year based indexes. FIXME: Ignoring leap years here
        caldata = xr.open_dataset(calfolder+crop+calsuf)
        # planarr = caldata[planvar]
        # harvarr = caldata[harvvar]
        # harvarr = xr.where(harvarr < planarr,harvarr + 365,harvarr) - 1
        planarr = caldata[planvar]
        harvarr = caldata[harvvar]
        planarr = xr.where(planarr < 213, planarr + 365, planarr)
        harvarr = xr.where(harvarr < 213, harvarr + 365, harvarr)
        harvarr = xr.where((harvarr - planarr) < 90, planarr + 90,harvarr) - 1 
        harvarr.name = caldata[harvvar].name
        xr.merge([planarr,harvarr]).to_netcdf(outfolder+"aux.calendar."+".nc")
        

        #FIXME: Loop here
        # hyear = hyears[0]
        for hyear in tqdm.tqdm(hyears):
            # # These are for yearly calendars
            # planarr = caldata[planvar].sel(year = hyear)
            # harvarr = caldata[harvvar].sel(year = hyear)
            # harvarr = xr.where(harvarr < planarr,harvarr + 365,harvarr) - 1 

            # Open the climate arrays, concatenating them
            temparr = read_cfnoleap_year(infolder,temppref,hyear)
            tmaxarr = read_cfnoleap_year(infolder,tmaxpref,hyear)
            tminarr = read_cfnoleap_year(infolder,tminpref,hyear)
            precarr = read_cfnoleap_year(infolder,precpref,hyear)
            vpdarr  = read_cfnoleap_year(infolder,vpdpref,hyear)

            # Make sure the calendar arrays match the coordinates
            # FIXME: This could be done only once, but then we would have to read 
            # a climate array beforehand
            if not (planarr["lat"].equals(temparr["lat"]) and planarr["lon"].equals(temparr["lon"]) ):
                planarr = planarr.interp_like(temparr)
                harvarr = harvarr.interp_like(temparr)


            # Generate a vector of lower T bounds to use as metadata and speed up computation
            Tlos = np.arange(Tlo,Thi,Tint)

            # Preallocate the arrays that will be filled
            lldims = ("lat","lon")
            coords = [(i,temparr.coords[i].data,temparr.coords[i].attrs) for i in lldims] # Tuples with lat and lon dimension specs

            # 2D arrays
            tempmean = xr.DataArray(coords = coords, name = "tempmean")
            tmaxmean = xr.DataArray(coords = coords, name = "tmaxmean")
            tminmean = xr.DataArray(coords = coords, name = "tminmean")
            precmean = xr.DataArray(coords = coords, name = "precmean")
            vpdmean = xr.DataArray(coords = coords, name = "vpdmean")


            trngmean = xr.DataArray(coords = coords, name = "trngmean")

            ndayarr = xr.DataArray(coords = coords, name = "ndays")

            # 2D + tmp arrays
            tmp = ("tmp",Tlos,{"long_name":"Temperature interval lower bound","units":"degC"})
            coords3d = [tmp] + coords
            tempdist = xr.DataArray(np.nan, coords = coords3d, name = "tempdist")
            tempgdds = xr.DataArray(np.nan, coords = coords3d, name = "tempgdds")

            # This basically creates pointers to the numpy arrays inside the xr.Dataarrays
            # We need those for numba to work. An alternative would be passing the .data in the function call
            planmat = planarr.data
            harvmat = harvarr.data

            tempmat = temparr.data
            tmaxmat = tmaxarr.data
            tminmat = tminarr.data
            precmat = precarr.data
            vpdmat = vpdarr.data

            tempmeanmat = tempmean.data
            tmaxmeanmat = tmaxmean.data
            tminmeanmat = tminmean.data
            precmeanmat = precmean.data
            vpdmeanmat = vpdmean.data

            trngmeanmat = trngmean.data

            ndaymat = ndayarr.data

            tempdistmat = tempdist.data
            tempgddsmat = tempgdds.data


            # Calculates everything.
            # FIXME: In this version, tempdistmat doesn't get filled
            calc_all(planmat,harvmat,tempmat,tmaxmat,tminmat,precmat,
                    vpdmat,
                    tempmeanmat,tmaxmeanmat,tminmeanmat,precmeanmat,
                    vpdmeanmat,
                    trngmeanmat,ndaymat,
                    tempdistmat,tempgddsmat)

            
            # Merge everything in a single Dataset
            # outdata = xr.merge([tempmean,tmaxmean,tminmean,precmean,vpdmean,trngmean,ndayarr,tempdist,tempgdds])
            outdata = xr.merge([tempmean,tmaxmean,tminmean,precmean,vpdmean,trngmean,ndayarr,tempgdds])
            outdata.attrs['Crop'] = crop
            outdata.attrs['harvest_year'] = hyear
            outdata.attrs['calendar_path'] = calfolder+crop+calsuf
            outdata.attrs['climdata_path'] = infolder
            outdata.attrs['climdata_ex_path'] = infolder+temppref+str(hyear)+".nc"

            # Write output
            outfname = outfolder + crop + ".computed." + str(hyear) + ".nc"
    #         outdata.to_netcdf(outfname)
            outdata.to_netcdf(outfname,
                    engine = "netcdf4",
                    encoding = {"tempgdds":{'zlib': True, 'complevel': 1}} )


# %%



# %%


