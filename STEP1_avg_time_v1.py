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

basefolder = "/media/gabriel/hd2_6tb/sdumont/monthly_sdumont/"
baseoutfolder = "output/"

scens = ["rcp8.5_weg", "rcp8.5_seg", "rcp2.6_weg", "rcp2.6_seg"]

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
    infolders = [basefolder + i + "/run/" for i in os.listdir(basefolder) if regex.match(i) ]
    #%% Folders (members) loop
    # infolder = infolders[0]
    for infolder in infolders:

        memberstr = (os.path.split(os.path.split(os.path.split(infolder)[0])[0])[1]).split(sep = "_")[-1]

        # filenames = glob.glob(infolder + '*{2030..2050}*')
        filenames = [glob.glob(infolder + "*." + mon + ".*")  for mon in [str(year) + "-" + i for year in years for i in [str(j).zfill(2) for j in range(1,13)]]]
        filenames = [i for j in filenames for i in j] # Flatten it

        #%% Open a folder's (member) dataset
        inds = xr.open_mfdataset(filenames)

        # Means
        tmeands = inds.groupby("time.month").mean(keep_attrs = True)
        
        # Variances
        tvards = inds.groupby("time.month").var(keep_attrs = True, ddof = 1)
        tvards = tvards.rename({i:(i + "_var") for i in tvards.data_vars})

        toutds = xr.merge([tmeands,tvards])


        #%% Save output. Since we're lazy loading, this is where most of the processing happens
        outfname = outfolder + scen + "_" + memberstr + "_" + str(syear) + "_" + str(eyear) + ".nc"
        print(outfname + " | " + datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        toutds.to_netcdf(outfname)
