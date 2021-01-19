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
import nc_time_axis
# %matplotlib inline

#%%
# filename = "/media/gabriel/hd2_6tb/ccsm4/test/rcp0.0_veg_005.cam2.h0.1950-01.nc"

# basefolder = "/media/gabriel/hd2_6tb/sdumont/monthly_sdumont/"
basefolder = "refdata/historical/rainy_season/"
baseoutfolder = "refdata/historical/rainy_season/ensemble/"

# scens = ["rcp8.5_weg", "rcp8.5_seg", "rcp2.6_weg", "rcp2.6_seg"]
scens = ["historical_heg"]

# The name of the file inside each folder
infsuf = "rs_allyears_2.5_1990_2004.nc"
outfsuf = "rs_allyears_ens_2.5_1990_2004.nc"

syear = 1990
eyear = 2004

years = list(range(syear, eyear + 1))

#%% Scenario loop
# scen = scens[0]
for scen in scens:

    # Create output folder
    outfolder = baseoutfolder + scen + "/"
    os.makedirs(outfolder, exist_ok=True)

    regex = re.compile(scen + "_.*")
    infolders = [basefolder + i for i in os.listdir(basefolder) if regex.match(i) ]

    filenames = [i + "/" + infsuf for i in infolders]

    # We can just concatenate over members without worrying about member names    
    items = [xr.open_dataset(f).expand_dims("member") for f in filenames]

    # WARNING: We're getting some rounoff error in dimensions, should tweak tolerance if using for other application
    items = [item.reindex_like(items[0], method = "nearest", tolerance = 0.001) for item in items]
    
    inds = xr.combine_nested(items,  concat_dim="member", combine_attrs= "override")

    # Select only the desired years
    inds = inds.sel(time = slice(syear, eyear))

    # Means
    tmeands = inds.mean(dim = ["time", "member"], keep_attrs = True)
    
    # Variances
    tvards = inds.var(dim = ["time", "member"], keep_attrs = True, ddof = 1)
    tvards = tvards.rename({i:(i + "_var") for i in tvards.data_vars})

    toutds = xr.merge([tmeands,tvards])

    toutds.attrs['nobs_var'] = inds.dims['member']*inds.dims['time']

    #%% Save output. Since we're lazy loading, this is where most of the processing happens
    outfname = outfolder + outfsuf
    print(outfname + " | " + datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    toutds.to_netcdf(outfname)
