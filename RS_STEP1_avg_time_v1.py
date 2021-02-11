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
#%%
# filename = "/media/gabriel/hd2_6tb/ccsm4/test/rcp0.0_veg_005.cam2.h0.1950-01.nc"

# basefolder = "/media/gabriel/hd2_6tb/sdumont/monthly_sdumont/"
basefolder = "output_rainy_season/"
baseoutfolder = "output_rainy_season/ensemble/"

scens = ["rcp8.5_weg", "rcp8.5_seg", "rcp2.6_weg", "rcp2.6_seg"]

# The name of the file inside each folder
infsuf = "rs_allyears_2.5_2005_2049.nc"
outfsuf = "rs_allyears_ens_2.5_2005_2049.nc"

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
    infolders = [basefolder + i for i in os.listdir(basefolder) if regex.match(i) ]

    filenames = [i + "/" + infsuf for i in infolders]

    # We can just concatenate over members without worrying about member names    
    items = [xr.open_dataset(f).expand_dims("member") for f in filenames]
    inds = xr.combine_nested(items,  concat_dim="member", combine_attrs= "override")

    # Select only the desired years
    inds = inds.sel(time = slice(syear, eyear))

    # Means
    tmeands = inds.mean(dim = ["time", "member"], keep_attrs = True)
    
    # Variances
    tvards = inds.var(dim = ["time", "member"], keep_attrs = True, ddof = 1)
    tvards = tvards.rename({i:(i + "_var") for i in tvards.data_vars})

    toutds = xr.merge([tmeands,tvards])


    #%% Save output. Since we're lazy loading, this is where most of the processing happens
    outfname = outfolder + outfsuf
    print(outfname + " | " + datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    toutds.to_netcdf(outfname)
