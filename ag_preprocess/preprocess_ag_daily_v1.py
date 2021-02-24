# %%
# from IPython import get_ipython

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import os
import re
import tqdm
import copy


# xr.set_options(display_style='html')
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# %%
# infolder = "../input/rcp_daily/"
infolder = "../output_ag/ag_daily/"
outfolder = "../output_ag/preprocessed/"
varnamedict = {
    "TREFHT" : "temp",
    "TREFHTMX" : "tmax",
    "TREFHTMN" : "tmin",
    "QREFHT" : "sphum",
    "PRECT" : "prec"
}
tempvarnames = ["temp", "tmax", "tmin"]
precvarname = "prec"
sphumvarname = "sphum"
vpdvarname = "vpd"

keepnames = tempvarnames + [precvarname] + [sphumvarname] + [vpdvarname]

topofname = "topo.nc"
reffname = "refgrid.nc"

# models = ['BCC-CSM2-MR', 'CanESM5', 'EC-Earth3-Veg', 'GFDL-CM4', 'MRI-ESM2-0']

# scens = ['historical','ssp245','ssp585']
scens = os.listdir(infolder)

# Domain, crop to Brazil
minlat = -35
maxlat = 7
minlon = 287
maxlon = 328

#%%
# Longitude flipping functions
def get_lon_name(d):
    if ('longitude' in d.dims):
        lonname = 'longitude'
    elif ('lon' in d.dims):
        lonname = 'lon'
    else:
        raise
    return lonname

# Flipped here means that it goes from -180 to 180
def global_is_lon_flipped(d):
    lonname = get_lon_name(d)
    return not not np.any(d[lonname]<0.0).data # The double negation converts it to a python boolean

# Just works from a nonflipped (0:360) global grid
def global_lon_flip(d):
    lonname = get_lon_name(d)
    nlon = d[lonname].size
    dflip = copy.deepcopy(d.roll(lon=int(nlon/2),roll_coords=False))
    # dflip[lonname] = dflip[lonname] - 180.0
    dflip = dflip.assign_coords({lonname : dflip[lonname] - 180.0})
    dflip = dlip.assign_
    return dflip

# 
def global_lon_unflip(d):
    lonname = get_lon_name(d)
    nlon = d[lonname].size
    dflip = copy.deepcopy(d.roll(lon=int(nlon/2),roll_coords=False))
    # dflip[lonname] = dflip[lonname] + 180.0
    dflip = dflip.assign_coords({lonname : dflip[lonname] + 180.0})
    return dflip

def brazil_lon_unflip(d):
    dflip = copy.deepcopy(d)
    lonname = get_lon_name(d) 
    # lons = d[lonname].data
    # dflip[lonname].data = lons-360
    dflip = dflip.assign_coords({lonname : dflip[lonname] - 360.0})
    return dflip

# %%

# Read topography and reference grid
topo = xr.open_dataarray(topofname)
refgrid = xr.open_dataset(reffname)
refgrid = refgrid[[i for i in refgrid.data_vars.keys()][0]]
try:
    refgrid = refgrid.isel(time = 0)
except: None



# allfiles = os.listdir(infolder)

invarnames = list(varnamedict.keys())

#%%

# scen = scens[0]
# scen = "historical"

for scen in scens:
    print(scen)

    # Dataset that will contain all variables
    # bigds = xr.open_mfdataset(infolder + "/" + scen + "/run/*.nc")[invarnames]
    bigds = xr.open_mfdataset(infolder + "/" + scen + "/*.nc")[invarnames]

    # Crop to domain
    bigds = bigds.sel(lat = slice(minlat,maxlat), lon = slice(minlon,maxlon))

    # Rename variables
    bigds = bigds.rename(varnamedict)

    # Temperature variables
    for tempvar in tempvarnames:
        meta = copy.deepcopy(bigds[tempvar].attrs)
        bigds[tempvar] = bigds[tempvar] - 273.15
        bigds[tempvar].attrs = meta
        bigds[tempvar].attrs["units"] = "degC"

    ## %%
    # Precipitation
    meta = copy.deepcopy(bigds[precvarname].attrs)
    bigds[precvarname] = bigds[precvarname] * 86400000.0
    bigds[precvarname].attrs = meta
    bigds[precvarname].attrs["units"] = "mm day-1"

    # %% VPD

    # Interpolate topography to generate a pressure field
    if global_is_lon_flipped(topo) != global_is_lon_flipped(bigds):
        topo = global_lon_unflip(topo)
    topo = topo.interp_like(bigds)
    pres = 1013.25*np.exp((-1)*(1.602769777072154)*np.log((np.exp(topo/10000.0)*213.15+75.0)/288.15))

    bigds["estmax"]=6.1078*np.exp((17.3*bigds["tmax"])/(237.3+bigds["tmax"]))
    bigds["estmin"]=6.1078*np.exp((17.3*bigds["tmin"])/(237.3+bigds["tmin"]))
    bigds["es"]=(bigds["estmax"]+bigds["estmin"])/2
    bigds["r"] = bigds["sphum"]/(1-bigds["sphum"])
    bigds["e"] = (bigds["r"]*pres)/(0.622+bigds["r"])
    bigds["rh"] = 100.0*bigds["e"]/bigds["es"]
    bigds[vpdvarname] = bigds["es"] - bigds["e"]
    bigds[vpdvarname] = bigds[vpdvarname].where(bigds[vpdvarname] >= 0,0) # Cap at zero

    bigds[vpdvarname].attrs["long_name"] = "Vapour pressure deficit"
    bigds[vpdvarname].attrs["units"] = "hPa"
    bigds[vpdvarname]

    # %%
    # Drop intermediate variables
    bigds = bigds.drop([i for i in bigds.data_vars.keys() if i not in keepnames])

    # # Rename coordinates to conform with Xavier
    # if ('lon' in bigds.dims) and ('lat' in bigds.dims):
    #     bigds = bigds.rename({'lon':'longitude', 'lat': 'latitude'}) # some models labelled dimensions differently...

    # # Unflip longitudes to be -180:180 like Xavier
    # bigds = brazil_lon_unflip(bigds)

    # # %%
    # outds = bigds.interp_like(refgrid)
    outds = bigds

    #%%
    print("Writing variables...")
    for varname in tqdm.tqdm(keepnames):
        scenoutfolder = outfolder + "/" + scen + "/"
        # Create output folder
        if not os.path.exists(scenoutfolder): os.makedirs(scenoutfolder, exist_ok=True)

        outfname = scenoutfolder + varname + ".daily.allyears.nc"
        outds[varname].to_netcdf(outfname)
    # %%