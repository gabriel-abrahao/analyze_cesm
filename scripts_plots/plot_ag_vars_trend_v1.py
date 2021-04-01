# Plots 
#%% 
import Ngl
import os
import re
import glob
import copy
from datetime import datetime
import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import plotnine as p9
# %matplotlib inline

sys.path.append("../")
import pooled_stats 

#%%
# REMEMBER PATHS ARE RELATIVE TO THE SCRIPT FOLDER!
motherfolder = "../"

crop = "single"
calname = "Oct15-Mar1_Jan15-Jun15"

baseinfolder = motherfolder + "output_ag/ag_vars/" + calname + "/" # Below here should be scenario folders, and below them the files
fsyear = 2012
feyear = 2050
infsuf = ".allscens.estimated.nc"

# Number of years to multiply the trend to
nyearstrend = 40

ifutdelta = False

cropstrdict = {"single" : "Soy", "maize" : "2nd crop Maize"}

# Plot deforestation X effects?
idefplots = True
if idefplots:
    definfolder = motherfolder + "input/out_surfdata_fix/" 
    defpyear   = 2050
    defrefyear = 2012
    
    # These should match the ones in the CESM run files. 
    defscenfnames = {'rcp2.6_cmp' : definfolder + "surfdata.pftdyn_0.9x1.25_rcp2.6_simyr1850-2100_c100323.nc",
                     'rcp2.6_seg' : definfolder + "surfdata.pftdyn_0.9x1.25_rcp2.6_simyr1850-2100_rochedo_seg_c100323.nc",\
                     'rcp2.6_weg' : definfolder + "surfdata.pftdyn_0.9x1.25_rcp2.6_simyr1850-2100_rochedo_weg_c100323.nc",\
                     'rcp8.5_cmp' : definfolder + "surfdata.pftdyn_0.9x1.25_rcp8.5_simyr1850-2100_c100319.nc",\
                     'rcp8.5_seg' : definfolder + "surfdata.pftdyn_0.9x1.25_rcp8.5_simyr1850-2100_rochedo_seg_c100319.nc",\
                     'rcp8.5_weg' : definfolder + "surfdata.pftdyn_0.9x1.25_rcp8.5_simyr1850-2100_rochedo_weg_c100319.nc"}

    defscennames = list(defscenfnames.keys())


    pftvarname = "PCT_PFT"

    # Indexes of PFTs considered as deforestation
    # FIXME: The actual Rochedo to CESM PFT calculation is more complex,
    # we have natural C4 grasses for example. This setup is only good for deltas
    defpfts = [14, 15, 16]
    # defpfts = [12, 13, 14, 15, 16] # This includes C3 grasses


# Read regions? We have to if we want to make all deforestation plots
ireadregions = True
regfname =      "../regions/states_grid.nc"
regcodesfname = "../regions/states_codes.csv"
useregs = ["AM","PA","MT","MA","TO","PI"]

# Contour variable
contvarname = "tempmean"
# contvarname = "agestimate"
# contvarname = "agestimate_perc"

# Select this model TODO: change the structure to support multi-model plots
usemodelstr = "GDD + EDD"
# usemodelstr = "Ensemble"

# String wih the type of overlay, or "none" for no overlay
overlaytype = "none"
# overlaytype = "wind"

# List of strings with where we should set non-significant points to NaN
# "over" means the overlay
# sigmodes = ["over","cont"]
sigmodes = ["cont"]
# sigmodes = []

# Significance level threshold
siglev = 0.05

# Add text to the figure mentioning the significance level. Override and don't add it if no sigmode
addsiglabel = True
if len(sigmodes) == 0:
    addsiglabel = False

# Domain string, limits are defined for each one further below
# domain = "BIG1"
# domain = "SAMATL" # South america and Atlantic Ocean
domain = "BR" # Zoom in Brazil

plotfname =     "../output_plots/ag_new/" + "/" + contvarname + "/" + "trend_" + "deltahist_" + usemodelstr + "_" + str(fsyear) + "_" + str(feyear) + "_" + domain + "_" + contvarname + "_sig_" + str((siglev*100)) + "pp" + "_".join(sigmodes) 
efplotfname =   "../output_plots/ag_new/" + "/" + contvarname + "/" + "trend_" + "effects_" + usemodelstr + "_" + str(fsyear) + "_" + str(feyear) + "_" + domain + "_" + contvarname + "_sig_" + str((siglev*100)) + "pp" + "_".join(sigmodes) 

defplotfname =   "../output_plots/ag_new/" + "/" + "deforestation" + "/" + "trend_" + "effects_" + str(fsyear) + "_" + str(feyear) + "_" + domain

wks_type = "png"

# ================= Variable and level dependent plotting parameters
# WARNING: Below are some predefined values of many of these 
# for specific variables which will override values in this section

# Default colormaps are red is higher. Reverse for things like pressure/height and precipitation
reversecolormap = False 
reversedeltacolormap = True

# Override the cnFillPallete resource in the level plots 
icolormapoverride   = True
colormapoverride    = "WhiteBlue"

# Levels
# contlevels  = np.arange(1300,1600,25)
deltalevels = np.arange(-4,4.1,.5)

# ===================== Predefined configurations for some variables

# # tempmean
if contvarname == "tempmean":
    contlevels  = np.arange(293,303,2)
    deltalevels = np.arange(-4,4.1,.5)
    # eflevels = np.arange(-3,3.1,0.25)
    eflevels = np.arange(-2,2.1,0.25)
    if ifutdelta:
        deltalevels = deltalevels/2.0
        eflevels = eflevels/2.0
    reversecolormap = False 
    reversedeltacolormap = False
    icolormapoverride   = False
# # agestimate
elif contvarname == "agestimate_perc":
    contlevels  = np.arange(0,10,1)
    deltalevels = np.arange(-30,30.1,5)
    eflevels = np.arange(-18,18.1,3)
    if ifutdelta:
        deltalevels = deltalevels/2.0
        eflevels = np.arange(-12,12.1,2)
    reversecolormap = False 
    reversedeltacolormap = True
    icolormapoverride   = False
    colormapoverride    = "WhiteBlue"
    idropwhites = True
# # PRECT
elif contvarname == "PRECT":
    contlevels  = np.arange(0,10,1)
    deltalevels = np.arange(-3,3.1,0.5)
    eflevels = np.arange(-2,2.1,0.25)
    reversecolormap = False 
    reversedeltacolormap = True
    icolormapoverride   = True
    colormapoverride    = "WhiteBlue"
# # TREFHT
elif contvarname == "TREFHT":
    contlevels  = np.arange(293,303,2)
    deltalevels = np.arange(-4,4.1,.5)
    # eflevels = np.arange(-3,3.1,0.25)
    eflevels = np.arange(-2,2.1,0.25)
    reversecolormap = False 
    reversedeltacolormap = False
    icolormapoverride   = False
# # outslen
elif contvarname == "outslen":
    contlevels  = np.arange(100,250,10)
    deltalevels = np.arange(-20,20.1,)
    # eflevels = np.arange(-3,3.1,0.25)
    eflevels = np.arange(-15,15.1,1)
    reversecolormap = False 
    reversedeltacolormap = True
    icolormapoverride   = True
    colormapoverride    = "WhiteBlue"
# # outoday
elif contvarname == "outoday":
    contlevels  = np.arange(30,150,10)
    deltalevels = np.arange(-20,20.1,)
    # eflevels = np.arange(-3,3.1,0.25)
    eflevels = np.arange(-15,15.1,1)
    reversecolormap = False 
    reversedeltacolormap = False
    icolormapoverride   = False
    colormapoverride    = "WhiteBlue"
# # outeday
elif contvarname == "outeday":
    contlevels  = np.arange(240,360,10)
    deltalevels = np.arange(-20,20.1,)
    # eflevels = np.arange(-3,3.1,0.25)
    eflevels = np.arange(-15,15.1,1)
    reversecolormap = True 
    reversedeltacolormap = True
    icolormapoverride   = False
    colormapoverride    = "WhiteBlue"

else:
    print("WARNING: No preset for variable " + contvarname)


#%%
# Derived configurations

# Remove the 0 in the middle of deltalevels, so the white spans both positive and negative sides
# Do the same for eflevels, which are also anomalies
if "cont" not in sigmodes:
    deltalevels = np.delete(deltalevels,np.where(deltalevels == 0))
    eflevels = np.delete(eflevels,np.where(eflevels == 0))

# ==================================================== Domains
if domain == "BIG1":
    minlat = -61
    maxlat = 50
    minlon = 230
    maxlon = 360
# South America and part of the adjoining Atlantic Ocean
if domain == "SAMATL":
    minlat = -61
    maxlat = 15
    minlon = 273
    maxlon = 345
# Brazil only
if domain == "BR":
    minlat = -35
    maxlat = 7
    minlon = 285
    maxlon = 328

# ================================================== Unit conversions
# List of all variables that need conversion or metadata fixing
# Also insert appropriate long_name and unit attributes as needed
convvarnames = ["PRECT", "outslen", "outoday", "outeday"]
if contvarname == "PRECT":
    contlongname = "Total Precipitation"
    unitfrom = "m/s"
    unitto = "mm/day"
    convfac = 86400000.0 # m/s to mm/day
    convsum = 0.0
elif contvarname == "outslen":
    contlongname = "Rainy season length"
    unitfrom = "n/a"
    unitto = "days"
    convfac = 1.0 # unchanged
    convsum = 0.0
elif contvarname == "outoday":
    contlongname = "Rainy season onset day"
    unitfrom = "n/a"
    unitto = "days since Jul 1"
    convfac = 1.0 # unchanged
    convsum = 0.0
elif contvarname == "outeday":
    contlongname = "Rainy season end day"
    unitfrom = "n/a"
    unitto = "days since Jul 1"
    convfac = 1.0 # unchanged
    convsum = 0.0


# ================================================== Variables
# Select these variables for performing t-tests. xr.apply_ufunc
# is apparently not good at lazy loading, and will calculate
# t-tests for all variables in the datasets
if overlaytype == "none":
    overvars = []
elif overlaytype == "wind":
    overvars = ["U","V"]



# %% Self setups

# Create the output folder
os.makedirs(os.path.dirname(plotfname), exist_ok=True)

# Set up a bounding box right outside the plotting area so the wind vectors are nice
rminlat = minlat-3
rmaxlat = maxlat+3
rminlon = minlon-3
rmaxlon = maxlon+3

# Month numbers to strings
monstrs = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",\
    5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",\
        10:"Oct",11:"Nov",12:"Dec"}

#%% Function definitions

# Add a suffix to all variables in a Dataset
def add_suf(ds,suf):
    return(ds.rename({i:(i + suf) for i in ds.data_vars}))

# Removes a suffix from all variables in a Dataset
def rem_suf(ds,suf):
    return(ds.rename({i:(re.sub("(.*)"+suf+"$","\g<1>",i)) for i in ds.data_vars}))

# Add a suffix to all strings in a list
def add_suf_str(strlst,suf):
    return([i + suf for i in strlst])

# Removes a suffix to all strings in a list
def rem_suf_str(strlst,suf):
    return([re.sub("(.*)"+suf+"$","\g<1>",i) for i in strlst])

# Filters a string based on a suffix
def filter_suf_str(strlst,suf, keep = "in"):
    rx = re.compile("(.*)"+suf+"$")
    if keep == "in":
        output = [i for i in strlst if rx.match(i) is not None]
    elif keep == "out":
        output = [i for i in strlst if rx.match(i) is None]
    else:
        raise NameError("Parameter keep must be 'in' or 'out', got " + str(keep))
    return(output)

# Adds metadata to several variables based on a dictionary
def add_meta_dict(ds,metadict):
    for varname in metadict.keys():
        if varname in ds.data_vars:
            ds[varname] = ds[varname].assign_attrs(metadict[varname])
    return(ds)

# Given a list of strings, remove the suffix suf from all
# elements but also return a boolean mask list of which
# elements were substituted
def str_suf_rem_and_mask(strlst, suf):
    rx = re.compile("(.*)"+suf+"$")
    tuplist = [(rx.sub("\g<1>",i), rx.match(i) is not None) for i in strlst]
    return(tuple([list(i) for i in  zip(*tuplist)]))

# Splits a dataset that has means and variances coded by name. 
# e.g. the mean of variable X is in variable X and it's variance is in X_val
def split_dataset_variances(dsboth, selvars, uselev):
    regexvar = re.compile(".*_var$")
    dsmeans = dsboth[[i for i in dsboth.data_vars if not regexvar.match(i)]]
    dsvariances = dsboth[[i for i in dsboth.data_vars if regexvar.match(i)]]
    dsvariances = dsvariances.rename({i:(re.sub("(.*)_var$","\g<1>",i)) for i in dsvariances.data_vars})

    # Select just the desired variables
    dsmeans = dsmeans[selvars]
    dsvariances = dsvariances[selvars]

    # If any of selvars has lev, select the single uselev
    # Using slice like this here ensures lev is kept as a singleton dimension
    if "lev" in dsmeans.coords:
        dsmeans = dsmeans.sel(lev=slice(uselev,uselev))
        dsvariances = dsvariances.sel(lev=slice(uselev,uselev))

    # Add a singleton lev variable to single level variables
    # TODO: The order of dimensions is not kept the same here
    # Shouldn't be a problem though since we'll select a single 
    # slice in the main script
    for v in dsmeans.data_vars:
        if "lev" not in dsmeans[v].coords:
            dsmeans[v] = dsmeans[v].expand_dims("lev").assign_coords({"lev" : [uselev]})
            dsvariances[v] = dsvariances[v].expand_dims("lev").assign_coords({"lev" : [uselev]})


    # Using slice like this here ensures lev is kept as a singleton dimension
    # dsmeans = dsmeans[selvars].sel(lev=slice(uselev,uselev))
    # dsvariances = dsvariances[selvars].sel(lev=slice(uselev,uselev))
    return((dsmeans, dsvariances))

# Calculates the difference between "contvarname" and "overvars" in a dataset
# and also returns a "contvarname"_pval variable with the p-values of a
# t-test on the difference of two means given a "contvarname"_var variable
def calc_diff_ttest(dsboth1, dsboth2, selvars, sigmodes, nobs):
    selvars_both = selvars + ([i + "_var" for i in selvars])

    (dsmeans1, dsvariances1) = pooled_stats.split_dataset_variances_generic(dsboth1[selvars_both])
    (dsmeans2, dsvariances2) = pooled_stats.split_dataset_variances_generic(dsboth2[selvars_both])

    diff = dsmeans1 - dsmeans2

    if len(sigmodes) == 0:
        return(diff)
    else:
        # t-test
        testvarnames = [contvarname]
        if "over" in sigmodes:
            testvarnames.extend(overvars)

        dsttest = xr.apply_ufunc(
                ttest_ind_from_stats,
                dsmeans1[testvarnames],
                dsvariances1[testvarnames]**0.5,
                nobs,
                dsmeans2[testvarnames],
                dsvariances2[testvarnames]**0.5,
                nobs,
                True,
                input_core_dims=[[], [], [], [], [], [], []],
                output_core_dims=[[], []],
                vectorize=True,
                # keep_attrs=True,
                dask='parallelized',
            )[1]

        dsttest = dsttest.rename({i:(i + "_pval") for i in dsttest.data_vars})
        
        # Variance of the difference
        dsvariances = dsvariances1 + dsvariances2
        dsvariances = dsvariances.rename({i:(i + "_var") for i in dsvariances.data_vars})
        
        diff = diff.merge(dsttest)
        diff = diff.merge(dsvariances)
        # diff.expand_dims("lev")
        # diff["lev"] = np.array(uselev)
        # (diff,dump) = xr.broadcast(diff, dsboth1)
        return(diff)

# Calculates a t-test of the difference to a constant meanpop
def calc_diff_ttest_1samp(dsboth1, meanpop, selvars, sigmodes, nobs):

    selvars_both = selvars + ([i + "_var" for i in selvars])

    (dsmeans1, dsvariances1) = pooled_stats.split_dataset_variances_generic(dsboth1[selvars_both])

    diff = dsmeans1 - meanpop
    diff.attrs = dsmeans1.attrs
    diff = add_meta_dict(diff, {i:dsmeans1[i].attrs for i in dsmeans1.data_vars})

    if len(sigmodes) == 0:
        return(diff)
    else:
        # t-test
        testvarnames = selvars

        dsttest = xr.apply_ufunc(
                ttest_ind_from_stats,
                dsmeans1[testvarnames],
                dsvariances1[testvarnames]**0.5,
                nobs,
                meanpop,
                0.0,
                nobs,
                True,
                input_core_dims=[[], [], [], [], [], [], []],
                output_core_dims=[[], []],
                vectorize=True,
                # keep_attrs=True,
                dask='parallelized',
            )[1]

        dsttest = dsttest.rename({i:(i + "_pval") for i in dsttest.data_vars})
        
        # Variance of the difference (unchanged)
        dsvariances = dsvariances1
        dsvariances = dsvariances.rename({i:(i + "_var") for i in dsvariances.data_vars})
        
        diff = diff.merge(dsttest)
        diff = diff.merge(dsvariances)
        # diff.expand_dims("lev")
        # diff["lev"] = np.array(uselev)
        # (diff,dump) = xr.broadcast(diff, dsboth1)
        return(diff)


# poi = calc_diff_ttest(bigdsin.sel(scenario = "rcp2.6_weg"), bigdsin.sel(scenario = "rcp2.6_seg"), contvarname, overvars, uselev, nobs)
# poi
# dsboth1 = bigdsin.sel(scenario = "rcp2.6_weg")
# dsboth2 = bigdsin.sel(scenario = "rcp2.6_seg")

def reset_coord_resources_ref(res, ds):
    res.sfXArray               = ds["lon"].values
    res.sfYArray               = ds["lat"].values

    res.vfXArray               = ds["lon"].values
    res.vfYArray               = ds["lat"].values
    return(res)


# def lintrans_means_ds(ds, factor)


#%%
# ============================================= BEGIN READING FILES

deltafname = baseinfolder + "/" + crop + ".trend." + str(fsyear) + "-" + str(feyear) + infsuf
# deltafname = "/media/gabriel/hd1_6tb/backup/gabriel/transicao/doutorado/analyze_cesm/output_ag/ag_vars/Oct15-Mar1_Jan15-Jun15/poi.nc"
deltads = xr.open_dataset(deltafname)

# Get historical period and build strings
hsyear = deltads.attrs["perc_base_syear"]
heyear = deltads.attrs["perc_base_eyear"]
histimestr = "(" + str(hsyear) + "-" + str(heyear) + ")"
futtimestr = "(" + str(fsyear) + "-" + str(feyear) + ")"

# Reorder scenarios
usescens = ['rcp2.6_cmp', 'rcp2.6_seg', 'rcp2.6_weg', 'rcp8.5_cmp', 'rcp8.5_seg', 'rcp8.5_weg']
deltads = deltads.sel(scenario = usescens)

#%%
# Calculate significance of the trend using a t-test against 0
deltads = calc_diff_ttest_1samp(deltads, 0.0, [contvarname]+overvars, sigmodes, deltads.attrs["nyears"]-2)

# Comparing all EG scenarios to their CMIP reference
# We do this here before converting units
efds26 = calc_diff_ttest(deltads.sel(scenario = ["rcp2.6_seg","rcp2.6_weg"]), deltads.sel(scenario = ["rcp2.6_cmp"]).squeeze("scenario", drop = True), [contvarname]+overvars, sigmodes, deltads.attrs["nyears"]-2)
efds85 = calc_diff_ttest(deltads.sel(scenario = ["rcp8.5_seg","rcp8.5_weg"]), deltads.sel(scenario = ["rcp8.5_cmp"]).squeeze("scenario", drop = True), [contvarname]+overvars, sigmodes, deltads.attrs["nyears"]-2)

efds = xr.merge([efds26,efds85])

#%% Multiply trend to number of years and fix metadata after t-tests 
deltads[contvarname] = deltads[contvarname]*nyearstrend
efds[contvarname] = efds[contvarname]*nyearstrend

if contvarname == "agestimate_perc":    
    deltads[contvarname].attrs["long_name"] = cropstrdict[crop] + " yield trend over "+str(nyearstrend)+" years"
    deltads[contvarname].attrs["units"] = "%"

    efds[contvarname].attrs["long_name"] = "Difference in " + cropstrdict[crop] + " yield trend over "+str(nyearstrend)+" years"
    efds[contvarname].attrs["units"] = "p.p."
elif contvarname == "tempmean":
    deltads[contvarname].attrs["long_name"] = "Temperature trend over "+str(nyearstrend)+" years"
    deltads[contvarname].attrs["units"] = "degC"

    efds[contvarname].attrs["long_name"] = "Difference in " + "Temperature trend over "+str(nyearstrend)+" years"
    efds[contvarname].attrs["units"] = "degC"


# %%
# Convert units in the contour variable only if needed
# and add a long_name attribute if we don't have one
if contvarname in convvarnames:
    bigdsin[contvarname] = bigdsin[contvarname]*convfac + convsum
    bigdsin[contvarname].attrs["units"] = unitto
    bigdsin[contvarname].attrs["long_name"] = contlongname
    refds[contvarname] = refds[contvarname]*convfac + convsum
    refds[contvarname].attrs["units"] = unitto
    refds[contvarname].attrs["long_name"] = contlongname
    # Do the same for the variances if applicable
    contvarnamevariance = contvarname + "_var" 
    if contvarnamevariance in bigdsin.data_vars:
        bigdsin[contvarnamevariance] = bigdsin[contvarnamevariance]*convfac**2 + convsum
        bigdsin[contvarnamevariance].attrs["units"] = unitto
        bigdsin[contvarnamevariance].attrs["long_name"] = contlongname
    if contvarnamevariance in refds.data_vars:
        refds[contvarnamevariance] = refds[contvarnamevariance]*convfac**2 + convsum
        refds[contvarnamevariance].attrs["units"] = unitto
        refds[contvarnamevariance].attrs["long_name"] = contlongname

#%% Read deforestation maps if asked
if idefplots:
    defallitems = [{"scenario" : i, "ds" : xr.open_dataset(defscenfnames[i]).\
        sel(lat = slice(rminlat,rmaxlat), lon = slice(rminlon,rmaxlon))} for i in defscennames]
    for item in defallitems:
        item["ds"] = item["ds"].expand_dims("scenario")
        item["ds"]["scenario"] = pd.Index([item["scenario"]])

    defbigdsin = xr.combine_nested([i["ds"] for i in defallitems], concat_dim="scenario", combine_attrs= "override")

    # Calculate deforestation
    defarr = defbigdsin.isel(pft = defpfts).sum(dim = "pft")[pftvarname]

    defarr = defarr.sel(time = defpyear) - defarr.sel(time = defrefyear)

    # Interpolate to match exactly the CESM grid. CESM itself is apparently far more lenient on coordinate matching than xarray.
    defarr = defarr.interp_like(deltads)    

    #TODO: This will only be valid in Brazil, since LU is different outside of it in each RCP.
    deltadefarr = defarr.sel(scenario = "rcp8.5_weg") - defarr.sel(scenario = "rcp8.5_seg")

    # cmpdeltadefarr = defarr.sel(scenario = "rcp8.5_weg") - defarr.sel(scenario = "rcp8.5_seg")

    # Subtract all DEF scenarios to their CMIP reference

    cmpdefdelta26 = defarr.sel(scenario = ["rcp2.6_seg","rcp2.6_weg"]) - defarr.sel(scenario = ["rcp2.6_cmp"]).squeeze("scenario", drop = True)
    cmpdefdelta85 = defarr.sel(scenario = ["rcp8.5_seg","rcp8.5_weg"]) - defarr.sel(scenario = ["rcp8.5_cmp"]).squeeze("scenario", drop = True)
    cmpdefdelta = xr.merge([cmpdefdelta26,cmpdefdelta85])
    cmpdefdelta = cmpdefdelta[pftvarname]

    # Rename
    deltadefarr = deltadefarr.rename("DiffDef")
    cmpdefdelta = cmpdefdelta.rename("DiffDef")

#%% Read regions
if ireadregions:
    regions = xr.open_dataarray(regfname).interp_like(deltads, method = "nearest").rename("region")

regcodes = pd.read_csv(regcodesfname).set_index("code").T.to_dict("list")
regcodes = {i:regcodes[i][0] for i in regcodes.keys()}

# %% ===================== BEGIN PLOTS ==============================================
slabelstring = deltads[contvarname].attrs["long_name"] + \
    " (" + deltads[contvarname].attrs["units"] + ") ~C~" + futtimestr + " relative to " + histimestr + ""

wksres = Ngl.Resources()
wksres.wkHeight = 2048
wksres.wkWidth = 2048
wks = Ngl.open_wks(wks_type,plotfname,wksres)  # Open a workstation.

# Here we coarsen the wind array so the map doesn't get cramped
# deltads = deltads.coarsen(lat = 2, lon = 2, boundary = "trim").mean()

# Function to set common resources
def set_common_resources():
    res = Ngl.Resources()

    res.nglDraw         =   False
    res.nglFrame         =   False

    res.sfXArray               = deltads["lon"].values
    res.sfYArray               = deltads["lat"].values

    res.vfXArray               = deltads["lon"].values
    res.vfYArray               = deltads["lat"].values

    res.mpLimitMode     = "LatLon"  # Change the area of the map
    res.mpMinLatF       =  minlat   # viewed.
    res.mpMaxLatF       =  maxlat  
    res.mpMinLonF       =  minlon
    res.mpMaxLonF       =  maxlon

    res.tmXBOn          =   False
    res.tmYLOn          =   False

    res.mpGridAndLimbOn =   False
    # res.tmXBBorderOn    =   False
    # res.tmXTBorderOn    =   False
    # res.tmYLBorderOn    =   False
    # res.tmYRBorderOn    =   False

    res.mpGeophysicalLineThicknessF = 3.0
    res.mpProvincialLineThicknessF  = 2.0
    res.mpDataSetName         = "Earth..4"
    res.mpDataBaseVersion     = "MediumRes"
    res.mpOutlineSpecifiers   = "Brazil:states"

    return(res)


# Resources for reference contour/fill plots
scontres = set_common_resources()

scontres.cnFillOn               = True
scontres.cnLinesOn              = False
scontres.cnLineLabelsOn         = False
scontres.cnFillMode             = "RasterFill"

scontres.lbLabelBarOn           = False

scolormap = Ngl.read_colormap_file("BlueWhiteOrangeRed")
# if idropwhites:
#     scolormap = scolormap[(scolormap.sum(axis=1) != 4),:]
if reversedeltacolormap:
    scolormap = scolormap[::-1]

# scontres.cnFillPalette           =   "BlueWhiteOrangeRed"
scontres.cnFillPalette           =   scolormap
scontres.cnLevelSelectionMode    =   "ExplicitLevels"
scontres.cnLevels    =   deltalevels

# Override
if icolormapoverride:
    scontres.cnFillPalette           =   colormapoverride


# Scenario Plots
splots = []
sfigstrs = []
scenarios = deltads.scenario.values.tolist()
# for scen in scenarios:
#     for usemon in usemons:
for scen in scenarios:
    # contres.tiMainString = usemon
    seldict = {"scenario":scen}
    if "statmodel" in deltads.coords:
        seldict["statmodel"] = usemodelstr

    dssubset = deltads.sel(seldict)    
    # dssubset = deltads.sel(scenario = scen, statmodel = usemodelstr)

    if "cont" in sigmodes:
        dssubset[contvarname] = dssubset[contvarname].where(dssubset[contvarname+"_pval"]<=siglev,np.nan)

    contplot = Ngl.contour_map(wks,dssubset[contvarname].to_masked_array(),scontres)
    
    splots.append(contplot)
    sfigstrs.append(str(scen))

spanelres                                  = Ngl.Resources()
spanelres.nglPanelFigureStrings            = sfigstrs
spanelres.nglPanelFigureStringsFontHeightF = 0.01
spanelres.nglPanelLabelBar                 = True     # Turn on panel labelbar
# spanelres.nglPanelLabelBar                 = True     # Turn on panel labelbar
spanelres.nglPanelLabelBarLabelFontHeightF = 0.01    # Labelbar font height
spanelres.nglPanelLabelBarHeightF          = 0.03   # Height of labelbar
# spanelres.nglPanelLabelBarWidthF           = 0.700    # Width of labelbar
# spanelres.nglPanelTop                      = 0.75
spanelres.nglPanelBottom                      = 0.05
# spanelres.nglPanelLeft                      = 0.2

# spanelres.nglMaximize = True
# spanelres.nglFrame = False

spanelres.lbTitleString          =   slabelstring
spanelres.lbTitlePosition          =   "Bottom"
spanelres.lbTitleFontHeightF          =   0.01
spanelres.lbJustification          = "TopCenter"

if addsiglabel:
    spanelres.lbTitleString = spanelres.lbTitleString + "~C~Showing differences significant at " + str(siglev*100) + "%"

Ngl.panel(wks,splots,[2,len(scenarios)/2],spanelres)

# print(Ngl.retrieve_colormap(wks))
# print(Ngl.read_colormap_file("BlueWhiteOrangeRed"))

Ngl.delete_wks(wks)



# %% ===================== EFFECTS PLOTS ===================================================================
#    =======================================================================================================
#    =======================================================================================================
eflabelstring = efds[contvarname].attrs["long_name"] + " relative to CMIP5"\
    " (" + efds[contvarname].attrs["units"] + ") ~C~" + futtimestr + " relative to " + histimestr + ""
if ifutdelta:
    eflabelstring = "Difference in " + efds[contvarname].attrs["long_name"] + " relative to CMIP5"\
        " (" + efds[contvarname].attrs["units"] + ") ~C~" + futtimestr + "-" + histimestr + ""


wksres = Ngl.Resources()
wksres.wkHeight = 2048
wksres.wkWidth = 2048
wks = Ngl.open_wks(wks_type,efplotfname,wksres)  # Open a workstation.

# Resources for reference contour/fill plots
efcontres = set_common_resources()

efcontres.cnFillOn               = True
efcontres.cnLinesOn              = False
efcontres.cnLineLabelsOn         = False
efcontres.cnFillMode             = "RasterFill"

efcontres.lbLabelBarOn           = False

efcontres.mpOutlineBoundarySets = "NoBoundaries"

scolormap = Ngl.read_colormap_file("BlueWhiteOrangeRed")

if reversedeltacolormap:
    scolormap = scolormap[::-1]

# efcontres.cnFillPalette           =   "BlueWhiteOrangeRed"
efcontres.cnFillPalette           =   scolormap
efcontres.cnLevelSelectionMode    =   "ExplicitLevels"
efcontres.cnLevels    =   eflevels


# Scenario Plots
efplots = []
effigstrs = []
scenarios = efds.scenario.values.tolist()
# for scen in scenarios:
#     for usemon in usemons:
for scen in scenarios:
    seldict = {"scenario":scen}
    if "statmodel" in efds.coords:
        seldict["statmodel"] = usemodelstr

    dssubset = efds.sel(seldict)    
    # dssubset = efds.sel(scenario = scen, statmodel = usemodelstr)

    if "cont" in sigmodes:
        dssubset[contvarname] = dssubset[contvarname].where(dssubset[contvarname+"_pval"]<=siglev,np.nan)

    contplot = Ngl.contour_map(wks,dssubset[contvarname].to_masked_array(),efcontres)
    
    efplots.append(contplot)
    effigstrs.append(str(scen))

efpanelres                                  = Ngl.Resources()
efpanelres.nglPanelFigureStrings            = effigstrs
efpanelres.nglPanelFigureStringsFontHeightF = 0.01
efpanelres.nglPanelLabelBar                 = True     # Turn on panel labelbar
# efpanelres.nglPanelLabelBar                 = True     # Turn on panel labelbar
efpanelres.nglPanelLabelBarLabelFontHeightF = 0.01    # Labelbar font height
efpanelres.nglPanelLabelBarHeightF          = 0.03   # Height of labelbar
# efpanelres.nglPanelLabelBarWidthF           = 0.700    # Width of labelbar
# efpanelres.nglPanelTop                      = 0.75
efpanelres.nglPanelBottom                      = 0.05
# efpanelres.nglPanelLeft                      = 0.2

# efpanelres.nglMaximize = True
# efpanelres.nglFrame = False

efpanelres.lbTitleString          =   eflabelstring
efpanelres.lbTitlePosition          =   "Bottom"
efpanelres.lbTitleFontHeightF          =   0.01
efpanelres.lbJustification          = "TopCenter"

if addsiglabel:
    efpanelres.lbTitleString = efpanelres.lbTitleString + "~C~Showing differences significant at " + str(siglev*100) + "%"

Ngl.panel(wks,efplots,[2,len(scenarios)/2],efpanelres)

# print(Ngl.retrieve_colormap(wks))
# print(Ngl.read_colormap_file("BlueWhiteOrangeRed"))

Ngl.delete_wks(wks)

#%% ==== DEFORESTATION PLOTS ================================================
if idefplots:
    wksres = Ngl.Resources()
    wksres.wkHeight = 2048
    wksres.wkWidth = 2048
    wks = Ngl.open_wks(wks_type,defplotfname,wksres)  # Open a workstation.
    os.makedirs(os.path.dirname(defplotfname), exist_ok=True)

    deforres = set_common_resources()
    deforres = reset_coord_resources_ref(deforres, cmpdefdelta)

    deforres.cnFillOn               = True
    deforres.cnLinesOn              = False
    deforres.cnLineLabelsOn         = False
    deforres.cnFillMode             = "RasterFill"

    deforlabelstring = "Difference in "+str(defrefyear)+"-"+str(defpyear)+" deforestation from RCP"

    deforres.lbLabelBarOn           = False
    # deforres.lbLabelBarOn           = True
    deforres.lbOrientation          = "horizontal"
    deforres.lbTitleString          = deforlabelstring

    # deforcolormap = Ngl.read_colormap_file("WhiteYellowOrangeRed")
    # deforcolormap = deforcolormap[::-1]

    # deforcolormap = Ngl.read_colormap_file("OceanLakeLandSnow")
    # deforcolormap = deforcolormap[65:-24]
    # deforcolormap[0] = [1,1,1,1]
    # deforres.cnFillPalette           =   deforcolormap
    # deforres.cnLevels    =   np.append(np.array([5]),np.arange(10,70.1,10))
    
    # deforres.cnLevelSelectionMode    =   "AutomaticLevels"
    deforres.cnLevelSelectionMode    =   "ExplicitLevels"
    
    deforcolormap = Ngl.read_colormap_file("MPL_BrBG")[::-1]
    deforres.cnFillPalette           =   deforcolormap
    deforres.cnLevels    =   np.array([ -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40])

    defplots = []
    deffigstrs = []
    scenarios = cmpdefdelta.scenario.values.tolist()
    # for scen in scenarios:
    #     for usemon in usemons:
    for scen in scenarios:
        # contres.tiMainString = usemon
        dssubset = cmpdefdelta.sel(scenario = scen)
        dssubset = dssubset.where(np.abs(dssubset) > 1e-12) # A low threshold to eliminate off-country values
        defplot = Ngl.contour_map(wks,dssubset.to_masked_array(),deforres)

        defplots.append(defplot)
        deffigstrs.append(str(scen))

    defpanelres                                  = Ngl.Resources()
    defpanelres.nglPanelFigureStrings            = deffigstrs
    defpanelres.nglPanelFigureStringsFontHeightF = 0.01
    defpanelres.nglPanelLabelBar                 = True     # Turn on panel labelbar
    defpanelres.nglPanelLabelBarLabelFontHeightF = 0.01    # Labelbar font height
    defpanelres.nglPanelLabelBarHeightF          = 0.03   # Height of labelbar

    defpanelres.nglPanelBottom                   = 0.05

    defpanelres.lbTitleString                    =   deforlabelstring
    defpanelres.lbTitlePosition                  =   "Bottom"
    defpanelres.lbTitleFontHeightF               =   0.01
    defpanelres.lbJustification                  = "TopCenter"


    Ngl.panel(wks,defplots,[2,len(scenarios)/2],defpanelres)

    # print(Ngl.retrieve_colormap(wks))
    # print(Ngl.read_colormap_file("BlueWhiteOrangeRed"))

    Ngl.delete_wks(wks)

