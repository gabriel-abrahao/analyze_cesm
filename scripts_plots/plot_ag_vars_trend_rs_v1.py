# Plots 
#%% 
import Ngl
import os
import re
import glob
import copy
import functools
from datetime import datetime
import xarray as xr
import numpy as np
import pandas as pd
import shapefile
from scipy.stats import ttest_ind_from_stats
from statsmodels.formula.api import ols
import scipy
import sys

import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import plotnine as p9
# %matplotlib inline

sys.path.append("../")
import pooled_stats 
import geo_utils

#%%
# REMEMBER PATHS ARE RELATIVE TO THE SCRIPT FOLDER!
motherfolder = "../"

crop = "single"
# crop = "maize"
calname = "Oct15-Mar1_Jan15-Jun15"

baseinfolder = motherfolder + "output_ag/ag_vars/" + calname + "/" # Below here should be scenario folders, and below them the files
fsyear = 2012
feyear = 2050
infsuf = ".allscens.estimated.nc"

# Number of years to multiply the trend to
nyearstrend = 39

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

# Read biome raster and shape?
ireadbio       =    True
iplotbio       =    True
iaddsumdef     =    True
landmaskfname  =    "../auxdata/landmask.nc"
brmaskfname  =    "../auxdata/brmask.nc"
biorastfname   =    "../auxdata/bioma.nc"
bioshpfname    =    "../auxdata/biomas_WGS84_simp.shp"
# bioshpfname    =    "../auxdata/biomas_WGS84.shp"

# Read regions? We have to if we want to make all deforestation plots
ireadregions = True
# regtypestr = "states"
# regfname =      "../regions/states_grid.nc"
# regcodesfname = "../regions/states_codes.csv"
# useregs = ["AM","PA","MT","MA","TO","PI"]
regtypestr = "biomes"
regfname =      "../regions/biomes_grid.nc"
regcodesfname = "../regions/biomes_codes.csv"
useregs = ["AMAZ","CERR"]
reglongnamesdict = {
    "AMAZ" : "Amazonia",
    "CERR" : "Cerrado"
}

# Read agricultural observed data
ireadag         = True
agfname         = "../regions/agdata.nc"
agyear          = 2016
if crop == "single":
    agareavarname = "sharea"
    agprodvarname = "stprod"
elif crop == "maize":
    agareavarname = "m2harea"
    agprodvarname = "m2tprod"

iplotag         =   True # Overridden if we are not plotting yields
agshpfname = "../auxdata/vec_"+agareavarname+"frac_2016_gt_1pp.shp"
    

# ====================================== Contour variable ====================================
# contvarname = "tempmean"
# contvarname = "edd"
# contvarname = "gdd"
# contvarname = "vpdmean"
# contvarname = "precmean"
# contvarname = "agestimate_perc"

contvarname = "outslen"

if contvarname == "agestimate_perc":
    iyield = True
else: 
    iyield = False

if not iyield:
    iplotag = False

isrs = False
if contvarname in ["outslen", "outoday", "outeday"]:
    isrs = True

varshortnamedict = {
    "agestimate_perc" : "yield",
    "tempmean" : "daily mean temperature",
    "edd" : "EDD",
    "gdd" : "GDD",
    "vpdmean" : "mean VPD",
    "precmean" : "mean daily precipitation",
    "outslen" : "Rainy season length",
    "outoday" : "Rainy season onset day",
    "outeday" : "Rainy season end day"
}

# Select this model TODO: change the structure to support multi-model plots
# usemodelstr = "GDD + EDD"
usemodelstr = "Ensemble"

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

# Confidence interval for plotting
cival = 0.95

# Add text to the figure mentioning the significance level. Override and don't add it if no sigmode
# addsiglabel = True
addsiglabel = False
if len(sigmodes) == 0:
    addsiglabel = False

# Dictionary of scenario strings
scenstrdict =   {'rcp2.6_cmp' : 'RCP2.6-CMIP',
                 'rcp2.6_seg' : 'RCP2.6-SEG',\
                 'rcp2.6_weg' : 'RCP2.6-WEG',\
                 'rcp8.5_cmp' : 'RCP8.5-CMIP',\
                 'rcp8.5_seg' : 'RCP8.5-SEG',\
                 'rcp8.5_weg' : 'RCP8.5-WEG'}


# Domain string, limits are defined for each one further below
# domain = "BIG1"
# domain = "SAMATL" # South america and Atlantic Ocean
domain = "BR" # Zoom in Brazil

varfolder = "../output_plots/ag_new/" + crop + "/" + contvarname 

plotfname =     varfolder + "/" + "trend_" + "deltahist_" + usemodelstr + "_" + str(fsyear) + "_" + str(feyear) + "_" + domain + "_" + contvarname + "_sig_" + str((siglev*100)) + "pp" + "_".join(sigmodes) 
efplotfname =   varfolder + "/" + "trend_" + "effects_" + usemodelstr + "_" + str(fsyear) + "_" + str(feyear) + "_" + domain + "_" + contvarname + "_sig_" + str((siglev*100)) + "pp" + "_".join(sigmodes) 
egplotfname =   varfolder + "/" + "trend_" + "egef_" + usemodelstr + "_" + str(fsyear) + "_" + str(feyear) + "_" + domain + "_" + contvarname + "_sig_" + str((siglev*100)) + "pp" + "_".join(sigmodes) 

defplotfname =   "../output_plots/ag_new/" + "/" + "deforestation" + "/" + "defdiff_trend_" + "effects_" + str(fsyear) + "_" + str(feyear) + "_" + domain
deflevplotfname =   "../output_plots/ag_new/" + "/" + "deforestation" + "/" + "deflev_trend_" + "effects_" + str(fsyear) + "_" + str(feyear) + "_" + domain

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
idropwhites = True
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
    idropwhites = True
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
# # edd
elif contvarname == "edd":
    contlevels  = np.arange(0,10,1)
    deltalevels = np.arange(-70,70.1,10)
    eflevels = np.arange(-40,40.1,5)
    reversecolormap = False 
    reversedeltacolormap = False
    icolormapoverride   = False
    idropwhites = True
# # gdd
elif contvarname == "gdd":
    contlevels  = np.arange(0,10,1)
    deltalevels = np.arange(-250,250.1,25)
    eflevels = np.arange(-100,100.1,20)
    reversecolormap = False 
    reversedeltacolormap = False
    icolormapoverride   = False
    idropwhites = True
# # vpdmean
elif contvarname == "vpdmean":
    contlevels  = np.arange(0,10,1)
    deltalevels = np.arange(-5,5.1,0.5)
    eflevels = np.arange(-4,4.1,0.5)
    reversecolormap = False 
    reversedeltacolormap = False
    icolormapoverride   = False
    idropwhites = True
# # precmean
elif contvarname == "precmean":
    contlevels  = np.arange(0,10,1)
    deltalevels = np.arange(-3,3.1,0.5)
    eflevels = np.arange(-2,2.1,0.25)
    reversecolormap = False 
    reversedeltacolormap = True
    icolormapoverride   = False
    # colormapoverride    = "WhiteBlue"
    idropwhites = True
elif contvarname == "outslen":
    contlevels  = np.arange(100,250,10)
    deltalevels = np.arange(-21,21.1,3)
    eflevels = np.arange(-15,15.1,1)
    reversecolormap = True 
    reversedeltacolormap = True
    icolormapoverride   = False
    idropwhites = True
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
rminlat = minlat-5
rmaxlat = maxlat+5
rminlon = minlon-5
rmaxlon = maxlon+5

# Month numbers to strings
monstrs = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",\
    5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",\
        10:"Oct",11:"Nov",12:"Dec"}

#%% Function definitions

# Add a suffix to all variables in a Dataset
def add_suf(ds,suf):
    return(ds.rename({i:(i + suf) for i in ds.data_vars}))

# Removes a suffix from all variables in a Dataset (or DataArray)
def rem_suf(ds,suf):
    # If input is a DataArray, make it a Dataset
    isds = False
    if isinstance(ds, xr.DataArray):
        isds = True
        vname = ds.name
        ds = ds.to_dataset()

    out = ds.rename({i:(re.sub("(.*)"+suf+"$","\g<1>",i)) for i in ds.data_vars})
    out = out[list(out.data_vars)[0]]
    return(out)
    # return(ds.rename({i:(re.sub("(.*)"+suf+"$","\g<1>",i)) for i in ds.data_vars}))

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

# Drop unused vars with suffixes
def drop_unused(ds):
    varnames = ds.data_vars
    varnames = filter_suf_str(varnames, "_pval", keep="out")
    varnames = filter_suf_str(varnames, "_var", keep="out")
    outds = ds[varnames]
    return(outds)

def read_shapefile(shpfname,**kwargs):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' 
    column holding the geometry information. This uses the pyshp
    package
    """
    sf_shape = shapefile.Reader(shpfname, **kwargs)

    fields = [x[0] for x in sf_shape.fields][1:]
    records = [y[:] for y in sf_shape.records()]
    #records = sf_shape.records()

    # Extracts a list of tuples from a shape (in a shapefile) containing multiple parts 
    def extract_multiparts(s):
        # Append the last index to split the parts
        partinds = s.parts
        partinds = np.append(partinds,len(s.points))
        out = [s.points[partinds[i]:partinds[i+1]-1] for i in range(len(partinds)-1)]
        return(out)

    # shps = [s.points for s in sf_shape.shapes()]
    shps = [extract_multiparts(s) for s in sf_shape.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df

def add_shapefile_df_polygons(wks,plotobj,shp,res=None):
    # tcoords = next(iter(shp["coords"]))
    for tcoords in shp["coords"]:
        for shape in tcoords:
            (tx, ty) = tuple(zip(*shape))
            Ngl.add_polygon(wks, plotobj,tx,ty,res)

    # for tcoords in shp["coords"]:
    #     (tx, ty) = tuple(zip(*tcoords))
    #     Ngl.add_polygon(wks, plotobj,tx,ty,res)

def wavg(group, avg_name, weight_name):
    """ http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    In rare instance, we may not have weights, so just return the mean. Customize this if your business case
    should return otherwise.
    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        # return d.mean()
        return np.nan

# Aggregates all numeric variables in a Dataframe by groupvars by calculating a weighted average
def aggregate_weighted(dfin, groupvars, wgtvarname):
    aggvars = [i for i in dfin.columns if i not in (groupvars + [wgtvarname])]
    aggvars = [i for i in aggvars if np.issubdtype(dfin.dtypes[i], np.number)]
    dumdflist = [dfin.groupby(groupvars).apply(wavg,vname,wgtvarname).rename(vname) for vname in aggvars]
    dumdf = functools.reduce(lambda x, y: pd.merge(x, y, on = groupvars), dumdflist)
    return(dumdf)

#%%
# ============================================= BEGIN READING FILES

deltafname = baseinfolder + "/" + crop + ".rs..trend.rs." + str(fsyear) + "-" + str(feyear) + infsuf
# deltafname = baseinfolder + "/" + crop + ".trend." + str(fsyear) + "-" + str(feyear) + infsuf
# deltafname = "/media/gabriel/hd1_6tb/backup/gabriel/transicao/doutorado/analyze_cesm/output_ag/ag_vars/Oct15-Mar1_Jan15-Jun15/poi.nc"
deltads = xr.open_dataset(deltafname, decode_timedelta=False)

# Read masks
landmask = xr.open_dataarray(landmaskfname).interp_like(deltads, method="nearest")
brmask = xr.open_dataarray(brmaskfname).interp_like(deltads, method="nearest")

# Get historical period and build strings
hsyear = deltads.attrs["perc_base_syear"]
heyear = deltads.attrs["perc_base_eyear"]
histimestr = "(" + str(hsyear) + "-" + str(heyear) + ")"
futtimestr = "(" + str(fsyear) + "-" + str(feyear) + ")"

# Reorder scenarios
usescens = ['rcp2.6_cmp', 'rcp2.6_seg', 'rcp2.6_weg', 'rcp8.5_cmp', 'rcp8.5_seg', 'rcp8.5_weg']
deltads = deltads.sel(scenario = usescens)

#%%
# Get bias and reference to build bias-corrected rainy season plots (after the t-tests)
# We actually need only the reference
# Also mask Brazil since the RS definition is not great outside it
if isrs:
    deltads = deltads.where(brmask==1) # Mask before we do anything else

    rsref = deltads[contvarname+"_ref"]
    rsbias = deltads[contvarname+"_bias"]

    rsref = rem_suf(rsref,"_ref")
    rsbias = rem_suf(rsbias,"_bias")
#%%
# Calculate significance of the trend using a t-test against 0
deltads = calc_diff_ttest_1samp(deltads, 0.0, [contvarname]+overvars, sigmodes, deltads.attrs["nyears"]-2)

# Comparing all EG scenarios to their CMIP reference
# We do this here before converting units
efds26 = calc_diff_ttest(deltads.sel(scenario = ["rcp2.6_seg","rcp2.6_weg"]), deltads.sel(scenario = ["rcp2.6_cmp"]).squeeze("scenario", drop = True), [contvarname]+overvars, sigmodes, deltads.attrs["nyears"]-2)
efds85 = calc_diff_ttest(deltads.sel(scenario = ["rcp8.5_seg","rcp8.5_weg"]), deltads.sel(scenario = ["rcp8.5_cmp"]).squeeze("scenario", drop = True), [contvarname]+overvars, sigmodes, deltads.attrs["nyears"]-2)
efds = xr.merge([efds26,efds85])

# Comparing between EG scenarios
# We do this here before converting units
egds26 = calc_diff_ttest(deltads.sel(scenario = ["rcp2.6_weg"]), deltads.sel(scenario = ["rcp2.6_seg"]).squeeze("scenario", drop = True), [contvarname]+overvars, sigmodes, deltads.attrs["nyears"]-2)
egds85 = calc_diff_ttest(deltads.sel(scenario = ["rcp8.5_weg"]), deltads.sel(scenario = ["rcp8.5_seg"]).squeeze("scenario", drop = True), [contvarname]+overvars, sigmodes, deltads.attrs["nyears"]-2)
egds = xr.merge([egds26,egds85])


#%% Multiply trend to number of years and fix metadata after t-tests 
deltads[contvarname] = deltads[contvarname]*nyearstrend
efds[contvarname] = efds[contvarname]*nyearstrend
egds[contvarname] = egds[contvarname]*nyearstrend

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
elif contvarname == "edd":
    deltads[contvarname].attrs["long_name"] = "EDD trend over "+str(nyearstrend)+" years"
    deltads[contvarname].attrs["units"] = "degC day"

    efds[contvarname].attrs["long_name"] = "Difference in " + "EDD trend over "+str(nyearstrend)+" years"
    efds[contvarname].attrs["units"] = "degC day"
elif contvarname == "gdd":
    deltads[contvarname].attrs["long_name"] = "GDD trend over "+str(nyearstrend)+" years"
    deltads[contvarname].attrs["units"] = "degC day"

    efds[contvarname].attrs["long_name"] = "Difference in " + "GDD trend over "+str(nyearstrend)+" years"
    efds[contvarname].attrs["units"] = "degC day"
elif contvarname == "vpdmean":
    deltads[contvarname].attrs["long_name"] = "VPD trend over "+str(nyearstrend)+" years"
    deltads[contvarname].attrs["units"] = "hPa"

    efds[contvarname].attrs["long_name"] = "Difference in " + "VPD trend over "+str(nyearstrend)+" years"
    efds[contvarname].attrs["units"] = "hPa"
elif contvarname == "precmean":
    deltads[contvarname].attrs["long_name"] = "Precipitation trend over "+str(nyearstrend)+" years"
    deltads[contvarname].attrs["units"] = "mm day-1"

    efds[contvarname].attrs["long_name"] = "Difference in " + "Precipitation trend over "+str(nyearstrend)+" years"
    efds[contvarname].attrs["units"] = "mm day-1"
elif contvarname == "outslen":
    deltads[contvarname].attrs["long_name"] = "RS length trend over "+str(nyearstrend)+" years"
    deltads[contvarname].attrs["units"] = "days"

    efds[contvarname].attrs["long_name"] = "Difference in " + "RS length trend over "+str(nyearstrend)+" years"
    efds[contvarname].attrs["units"] = "days"

# Same attributes for EG differences
egds[contvarname].attrs = efds[contvarname].attrs

#%%
# Calculate bias-corrected projections of the rainy season variables
if isrs:
    rsbcds = deltads[contvarname] + rsref
    rsbcds = rsbcds.to_dataset()
    if contvarname == "outslen":
        missmask = ~np.isnan(rsbcds)
        rsbcds = rsbcds.where(rsbcds >= 0,0)
        rsbcds = rsbcds.where(rsbcds <= 365,365)
        rsbcds = rsbcds.where(missmask)
        # rsbc = rsbc.where(brmask == 1)

# rsbc.isel(scenario=0).plot(levels = [0,150,200,250])
# rsbc.isel(scenario=0).plot(levels = [0,150,200,250], cmap = "RdBu")
# rsbc.plot(row = "scenario", levels = [0,150,200,250], cmap = "RdBu")


# %%
# # Convert units in the contour variable only if needed
# # and add a long_name attribute if we don't have one
# if contvarname in convvarnames:
#     bigdsin[contvarname] = bigdsin[contvarname]*convfac + convsum
#     bigdsin[contvarname].attrs["units"] = unitto
#     bigdsin[contvarname].attrs["long_name"] = contlongname
#     refds[contvarname] = refds[contvarname]*convfac + convsum
#     refds[contvarname].attrs["units"] = unitto
#     refds[contvarname].attrs["long_name"] = contlongname
#     # Do the same for the variances if applicable
#     contvarnamevariance = contvarname + "_var" 
#     if contvarnamevariance in bigdsin.data_vars:
#         bigdsin[contvarnamevariance] = bigdsin[contvarnamevariance]*convfac**2 + convsum
#         bigdsin[contvarnamevariance].attrs["units"] = unitto
#         bigdsin[contvarnamevariance].attrs["long_name"] = contlongname
#     if contvarnamevariance in refds.data_vars:
#         refds[contvarnamevariance] = refds[contvarnamevariance]*convfac**2 + convsum
#         refds[contvarnamevariance].attrs["units"] = unitto
#         refds[contvarnamevariance].attrs["long_name"] = contlongname

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

    # Subtract WEG and SEG
    egdefdelta26 = defarr.sel(scenario = ["rcp2.6_weg"]) - defarr.sel(scenario = ["rcp2.6_seg"]).squeeze("scenario", drop = True)
    egdefdelta85 = defarr.sel(scenario = ["rcp8.5_weg"]) - defarr.sel(scenario = ["rcp8.5_seg"]).squeeze("scenario", drop = True)
    egdefdelta = xr.merge([egdefdelta26,egdefdelta85])
    egdefdelta = egdefdelta[pftvarname]
    

    # Rename
    deltadefarr = deltadefarr.rename("DiffDef")
    cmpdefdelta = cmpdefdelta.rename("DiffDef")

#%% Read regions
if ireadregions:
    regions = xr.open_dataarray(regfname).interp_like(deltads, method = "nearest").rename("region")

if "time" in regions.coords:
    regions = regions.squeeze("time", drop=True)

regcodes = pd.read_csv(regcodesfname).set_index("code").T.to_dict("list")
regcodes = {i:regcodes[i][0] for i in regcodes.keys()}

#%% Read agricultural data if asked
if ireadag:
    agdsin = xr.open_dataset(agfname).interp_like(deltads)
    agshp = read_shapefile(agshpfname)

    if iplotag:
        agres = Ngl.Resources()
        agres.gsFillOpacityF = 0.0
        agres.gsEdgesOn = True
        agres.gsEdgeColor = "green"
        agres.gsEdgeThicknessF = 6.0


#%% Read biome information
if ireadbio:
    biorast = xr.open_dataarray(biorastfname).interp_like(deltads, method="nearest")
    biorast = biorast.squeeze("time",drop=True)
    biorast = biorast.where(biorast != 0).rename("biome")

    bioshp = read_shapefile(bioshpfname)
    bioshp = bioshp.loc[bioshp["biomestr"].isin(useregs)]

    if iplotbio:
        biores = Ngl.Resources()
        biores.gsFillOpacityF = 0.0
        biores.gsEdgesOn = True
        biores.gsEdgeColor = "darkgreen"
        biores.gsEdgeThicknessF = 5.0

    # Summarize deforestation data by biome
    sumdeflev = xr.merge([defarr,biorast])
    # sumdeflev = sumdeflev.weighted(geo_utils.get_pixel_areas(sumdeflev)) # Open issue #3937 on xarray 0.16.2: no GroupBy weights
    sumdeflev = sumdeflev.groupby("biome").mean().to_dataframe().reset_index()
    # sumdeflev = bioshp.drop("coords",axis=1).rename(columns={"CD_Bioma" : "biome"}).merge(sumdeflev.reset_index(), on="biome")
    sumdeflev["biomestr"] = sumdeflev["biome"].replace(regcodes)
    sumdeflev = sumdeflev.query("biomestr.isin(@useregs)")    

    sumdefdiff = xr.merge([cmpdefdelta,biorast])
    sumdefdiff = sumdefdiff.groupby("biome").mean().to_dataframe().reset_index()
    # sumdefdiff = bioshp.drop("coords",axis=1).rename(columns={"CD_Bioma" : "biome"}).merge(sumdefdiff.reset_index(), on="biome")
    sumdefdiff["biomestr"] = sumdefdiff["biome"].replace(regcodes)
    sumdefdiff = sumdefdiff.query("biomestr.isin(@useregs)")    

    sumdefegdiff = xr.merge([egdefdelta,biorast])
    sumdefegdiff = sumdefegdiff.groupby("biome").mean().to_dataframe().reset_index()
    # sumdefegdiff = bioshp.drop("coords",axis=1).rename(columns={"CD_Bioma" : "biome"}).merge(sumdefegdiff.reset_index(), on="biome")
    sumdefegdiff["biomestr"] = sumdefegdiff["biome"].replace(regcodes)
    sumdefegdiff = sumdefegdiff.query("biomestr.isin(@useregs)")    
    
# %% ===================== LEVEL PLOTS ===================================================================
#    =======================================================================================================
#    =======================================================================================================
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

    res.mpOutlineBoundarySets = "National"

    return(res)

def add_fig_label(wks, plot, figcount):
    restext = Ngl.Resources()
    restext.txFontHeightF = 0.05
    # textx = 325.5
    textx = 287.5
    texty = 4.0
    
    textstr = chr(65+figcount)
    Ngl.add_text(wks,plot,textstr,textx,texty,restext)

# Resources for reference contour/fill plots
scontres = set_common_resources()

scontres.cnFillOn               = True
scontres.cnLinesOn              = False
scontres.cnLineLabelsOn         = False
scontres.cnFillMode             = "RasterFill"

scontres.lbLabelBarOn           = False

scolormap = Ngl.read_colormap_file("BlueWhiteOrangeRed")
if idropwhites:
    scolormap = scolormap[(scolormap.sum(axis=1) != 4),:]
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

    if iplotag:
        add_shapefile_df_polygons(wks,contplot,agshp,agres)


    add_fig_label(wks, contplot, len(splots))
    
    splots.append(contplot)
    sfigstrs.append(str(scenstrdict[scen]))

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

if idropwhites:
    scolormap = scolormap[(scolormap.sum(axis=1) != 4),:]
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

    if iplotag:
        add_shapefile_df_polygons(wks,contplot,agshp,agres)

    add_fig_label(wks, contplot, len(efplots))

    efplots.append(contplot)
    effigstrs.append(str(scenstrdict[scen]))

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

# %% ===================== EG DIFFERENCES PLOTS ===================================================================
#    =======================================================================================================
#    =======================================================================================================
eglabelstring = egds[contvarname].attrs["long_name"] + " (WEG-SEG)"\
    " (" + egds[contvarname].attrs["units"] + ") ~C~" + futtimestr + " relative to " + histimestr + ""
if ifutdelta:
    eglabelstring = "Difference in " + egds[contvarname].attrs["long_name"] + " relative to CMIP5"\
        " (" + egds[contvarname].attrs["units"] + ") ~C~" + futtimestr + "-" + histimestr + ""

# Use same colors as effects for now, since we maybe won't use these
eglevels = eflevels

wksres = Ngl.Resources()
wksres.wkHeight = 2048
wksres.wkWidth = 2048
wks = Ngl.open_wks(wks_type,egplotfname,wksres)  # Open a workstation.

# Resources for regerence contour/fill plots
egcontres = set_common_resources()

egcontres.cnFillOn               = True
egcontres.cnLinesOn              = False
egcontres.cnLineLabelsOn         = False
egcontres.cnFillMode             = "RasterFill"

egcontres.lbLabelBarOn           = False

# egcontres.mpOutlineBoundarySets = "NoBoundaries"
# egcontres.mpOutlineBoundarySets = "National"

scolormap = Ngl.read_colormap_file("BlueWhiteOrangeRed")

if idropwhites:
    scolormap = scolormap[(scolormap.sum(axis=1) != 4),:]
if reversedeltacolormap:
    scolormap = scolormap[::-1]

# egcontres.cnFillPalette           =   "BlueWhiteOrangeRed"
egcontres.cnFillPalette           =   scolormap
egcontres.cnLevelSelectionMode    =   "ExplicitLevels"
egcontres.cnLevels    =   eglevels

# Scenario Plots
egplots = []
egfigstrs = []
scenarios = egds.scenario.values.tolist()
# for scen in scenarios:
#     for usemon in usemons:
for scen in scenarios:
    seldict = {"scenario":scen}
    if "statmodel" in egds.coords:
        seldict["statmodel"] = usemodelstr

    dssubset = egds.sel(seldict)    
    # dssubset = egds.sel(scenario = scen, statmodel = usemodelstr)

    if "cont" in sigmodes:
        dssubset[contvarname] = dssubset[contvarname].where(dssubset[contvarname+"_pval"]<=siglev,np.nan)

    contplot = Ngl.contour_map(wks,dssubset[contvarname].to_masked_array(),egcontres)

    if iplotag:
        add_shapefile_df_polygons(wks,contplot,agshp,agres)

    add_fig_label(wks, contplot, len(egplots))

    egplots.append(contplot)
    # egfigstrs.append(str(scenstrdict[scen]))
    egfigstrs.append(str(scenstrdict[scen]).split("-")[0])

egpanelres                                  = Ngl.Resources()
egpanelres.nglPanelFigureStrings            = egfigstrs
egpanelres.nglPanelFigureStringsFontHeightF = 0.01
egpanelres.nglPanelLabelBar                 = True     # Turn on panel labelbar
# egpanelres.nglPanelLabelBar                 = True     # Turn on panel labelbar
egpanelres.nglPanelLabelBarLabelFontHeightF = 0.01    # Labelbar font height
egpanelres.nglPanelLabelBarHeightF          = 0.03   # Height of labelbar
# egpanelres.nglPanelLabelBarWidthF           = 0.700    # Width of labelbar
# egpanelres.nglPanelTop                      = 0.75
egpanelres.nglPanelBottom                      = 0.05
# egpanelres.nglPanelLegt                      = 0.2

# egpanelres.nglMaximize = True
# egpanelres.nglFrame = False

egpanelres.lbTitleString          =   eglabelstring
egpanelres.lbTitlePosition          =   "Bottom"
egpanelres.lbTitleFontHeightF          =   0.01
egpanelres.lbJustification          = "TopCenter"

if addsiglabel:
    egpanelres.lbTitleString = egpanelres.lbTitleString + "~C~Showing differences significant at " + str(siglev*100) + "%"

# Ngl.panel(wks,egplots,[1,len(scenarios)],egpanelres)
Ngl.panel(wks,egplots,[len(scenarios),1],egpanelres)

# print(Ngl.retrieve_colormap(wks))
# print(Ngl.read_colormap_file("BlueWhiteOrangeRed"))

Ngl.delete_wks(wks)
# # %% ===================== RAINY SEASON BC PLOTS ===================================================================
# #    =======================================================================================================
# #    =======================================================================================================
# eflabelstring = efds[contvarname].attrs["long_name"] + " relative to CMIP5"\
#     " (" + efds[contvarname].attrs["units"] + ") ~C~" + futtimestr + " relative to " + histimestr + ""
# if ifutdelta:
#     eflabelstring = "Difference in " + efds[contvarname].attrs["long_name"] + " relative to CMIP5"\
#         " (" + efds[contvarname].attrs["units"] + ") ~C~" + futtimestr + "-" + histimestr + ""


# wksres = Ngl.Resources()
# wksres.wkHeight = 2048
# wksres.wkWidth = 2048
# wks = Ngl.open_wks(wks_type,efplotfname,wksres)  # Open a workstation.

# # Resources for reference contour/fill plots
# efcontres = set_common_resources()

# efcontres.cnFillOn               = True
# efcontres.cnLinesOn              = False
# efcontres.cnLineLabelsOn         = False
# efcontres.cnFillMode             = "RasterFill"

# efcontres.lbLabelBarOn           = False

# efcontres.mpOutlineBoundarySets = "NoBoundaries"

# scolormap = Ngl.read_colormap_file("BlueWhiteOrangeRed")

# if idropwhites:
#     scolormap = scolormap[(scolormap.sum(axis=1) != 4),:]
# if reversedeltacolormap:
#     scolormap = scolormap[::-1]

# # efcontres.cnFillPalette           =   "BlueWhiteOrangeRed"
# efcontres.cnFillPalette           =   scolormap
# efcontres.cnLevelSelectionMode    =   "ExplicitLevels"
# efcontres.cnLevels    =   eflevels


# # Scenario Plots
# efplots = []
# effigstrs = []
# scenarios = efds.scenario.values.tolist()
# # for scen in scenarios:
# #     for usemon in usemons:
# for scen in scenarios:
#     seldict = {"scenario":scen}
#     if "statmodel" in efds.coords:
#         seldict["statmodel"] = usemodelstr

#     dssubset = efds.sel(seldict)    
#     # dssubset = efds.sel(scenario = scen, statmodel = usemodelstr)

#     if "cont" in sigmodes:
#         dssubset[contvarname] = dssubset[contvarname].where(dssubset[contvarname+"_pval"]<=siglev,np.nan)

#     contplot = Ngl.contour_map(wks,dssubset[contvarname].to_masked_array(),efcontres)

#     if iplotag:
#         add_shapefile_df_polygons(wks,contplot,agshp,agres)

#     add_fig_label(wks, contplot, len(efplots))

#     efplots.append(contplot)
#     effigstrs.append(str(scenstrdict[scen]))

# efpanelres                                  = Ngl.Resources()
# efpanelres.nglPanelFigureStrings            = effigstrs
# efpanelres.nglPanelFigureStringsFontHeightF = 0.01
# efpanelres.nglPanelLabelBar                 = True     # Turn on panel labelbar
# # efpanelres.nglPanelLabelBar                 = True     # Turn on panel labelbar
# efpanelres.nglPanelLabelBarLabelFontHeightF = 0.01    # Labelbar font height
# efpanelres.nglPanelLabelBarHeightF          = 0.03   # Height of labelbar
# # efpanelres.nglPanelLabelBarWidthF           = 0.700    # Width of labelbar
# # efpanelres.nglPanelTop                      = 0.75
# efpanelres.nglPanelBottom                      = 0.05
# # efpanelres.nglPanelLeft                      = 0.2

# # efpanelres.nglMaximize = True
# # efpanelres.nglFrame = False

# efpanelres.lbTitleString          =   eflabelstring
# efpanelres.lbTitlePosition          =   "Bottom"
# efpanelres.lbTitleFontHeightF          =   0.01
# efpanelres.lbJustification          = "TopCenter"

# if addsiglabel:
#     efpanelres.lbTitleString = efpanelres.lbTitleString + "~C~Showing differences significant at " + str(siglev*100) + "%"

# Ngl.panel(wks,efplots,[2,len(scenarios)/2],efpanelres)

# # print(Ngl.retrieve_colormap(wks))
# # print(Ngl.read_colormap_file("BlueWhiteOrangeRed"))

# Ngl.delete_wks(wks)

#%% ==== DEFORESTATION CMIP DIFFERENCE PLOTS ================================================
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

    deforlabelstring = "Difference in "+str(defrefyear)+"-"+str(defpyear)+" natural vegetation loss from RCP (p.p.)"

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

        if iplotbio:
            add_shapefile_df_polygons(wks,defplot,bioshp, biores)

            if iaddsumdef:
                restext = Ngl.Resources()
                restext.txFontHeightF = 0.025
                textx = 322.5
                texty = -28

                defdict = sumdefdiff.query("scenario==@scen").set_index("biomestr").to_dict()["DiffDef"]
                textstr = "~C~".join("{} : {:.2f}".format(k, v) for k, v in defdict.items())
                
                Ngl.add_text(wks,defplot,textstr,textx,texty, restext)

        add_fig_label(wks, defplot, len(defplots))

        defplots.append(defplot)
        deffigstrs.append(str(scenstrdict[scen]))

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

#%% ==== DEFORESTATION TIME DIFFERENCE PLOTS ================================================
if idefplots:
    wksres = Ngl.Resources()
    wksres.wkHeight = 2048
    wksres.wkWidth = 2048
    wks = Ngl.open_wks(wks_type,deflevplotfname,wksres)  # Open a workstation.
    os.makedirs(os.path.dirname(deflevplotfname), exist_ok=True)

    deflevres = set_common_resources()
    deflevres = reset_coord_resources_ref(deflevres, cmpdefdelta)

    deflevres.cnFillOn               = True
    deflevres.cnLinesOn              = False
    deflevres.cnLineLabelsOn         = False
    deflevres.cnFillMode             = "RasterFill"

    deflevlabelstring = "Natural vegetation loss between "+str(defrefyear)+" and "+str(defpyear)+" (p.p.)"

    deflevres.lbLabelBarOn           = False
    # deflevres.lbLabelBarOn           = True
    deflevres.lbOrientation          = "horizontal"
    deflevres.lbTitleString          = deflevlabelstring

    # deforcolormap = Ngl.read_colormap_file("WhiteYellowOrangeRed")
    # deforcolormap = deforcolormap[::-1]

    # deforcolormap = Ngl.read_colormap_file("OceanLakeLandSnow")
    # deforcolormap = deforcolormap[65:-24]
    # deforcolormap[0] = [1,1,1,1]
    # deflevres.cnFillPalette           =   deforcolormap
    # deflevres.cnLevels    =   np.append(np.array([5]),np.arange(10,70.1,10))
    
    # deflevres.cnLevelSelectionMode    =   "AutomaticLevels"
    deflevres.cnLevelSelectionMode    =   "ExplicitLevels"
    
    deforcolormap = Ngl.read_colormap_file("MPL_BrBG")[::-1]
    deflevres.cnFillPalette           =   deforcolormap
    deflevres.cnLevels    =   np.array([ -50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])

    deflevplots = []
    deflevfigstrs = []
    scenarios = defarr.scenario.values.tolist()
    # for scen in scenarios:
    #     for usemon in usemons:
    for scen in scenarios:
        # contres.tiMainString = usemon
        dssubset = defarr.sel(scenario = scen)
        # dssubset = dssubset.where(np.abs(dssubset) > 1e-12) # A low threshold to eliminate off-country values
        dssubset = dssubset.where(np.invert(np.isnan(landmask)))
        deflevplot = Ngl.contour_map(wks,dssubset.to_masked_array(),deflevres)

        if iplotbio:
            add_shapefile_df_polygons(wks,deflevplot,bioshp, biores)

            if iaddsumdef:
                restext = Ngl.Resources()
                restext.txFontHeightF = 0.03
                textx = 321.5
                texty = -27

                defdict = sumdeflev.query("scenario==@scen").set_index("biomestr").to_dict()[pftvarname]
                textstr = "~C~".join("{} : {:.2f}".format(k, v) for k, v in defdict.items())
                
                Ngl.add_text(wks,deflevplot,textstr,textx,texty, restext)

        add_fig_label(wks, deflevplot, len(deflevplots))


        deflevplots.append(deflevplot)
        deflevfigstrs.append(str(scenstrdict[scen]))

    defpanelres                                  = Ngl.Resources()
    defpanelres.nglPanelFigureStrings            = deflevfigstrs
    defpanelres.nglPanelFigureStringsFontHeightF = 0.01
    defpanelres.nglPanelLabelBar                 = True     # Turn on panel labelbar
    defpanelres.nglPanelLabelBarLabelFontHeightF = 0.01    # Labelbar font height
    defpanelres.nglPanelLabelBarHeightF          = 0.03   # Height of labelbar

    defpanelres.nglPanelBottom                   = 0.05

    defpanelres.lbTitleString                    =   deflevlabelstring
    defpanelres.lbTitlePosition                  =   "Bottom"
    defpanelres.lbTitleFontHeightF               =   0.01
    defpanelres.lbJustification                  = "TopCenter"


    Ngl.panel(wks,deflevplots,[2,len(scenarios)/2],defpanelres)

    # print(Ngl.retrieve_colormap(wks))
    # print(Ngl.read_colormap_file("BlueWhiteOrangeRed"))

    Ngl.delete_wks(wks)

#%% =============================================================TABULAR=============================

# Calculate confidence intervals
ciloz = scipy.stats.norm.ppf(0.5-(cival/2))
ciupz = scipy.stats.norm.ppf(0.5+(cival/2))
deltads["cilo"] = deltads[contvarname] + ciloz*(deltads[contvarname+"_var"]**0.5)
deltads["ciup"] = deltads[contvarname] + ciupz*(deltads[contvarname+"_var"]**0.5)

# Merge all level datasets
dsall = drop_unused(deltads).merge(defarr)
dsall = dsall.merge(agdsin.sel(year=agyear, drop=True))
dsall = dsall.merge(regions)

dsall["scenario"] = [scenstrdict[i] for i in list(dsall["scenario"].data)]
# Convert to DataFrame
dfall = dsall.to_dataframe().dropna(how="all").reset_index()
dfall = dfall.replace({"region":regcodes})
dfall["region"] = dfall["region"].loc[dfall["region"].isin(useregs)]
dfall = dfall.replace({"region":reglongnamesdict})
# dfall["rcpscen"] = dfall["scenario"].str[:6]
# dfall["luscen"] = dfall["scenario"].str[:6]
dfall["rcpscen"] = dfall["scenario"].str[:6]
dfall["luscen"] = dfall["scenario"].str[7:]

# Do the same for EG difference dataset
egdsall = drop_unused(egds).merge(egdefdelta)
egdsall = egdsall.merge(agdsin.sel(year=agyear, drop=True))
egdsall = egdsall.merge(regions)
egdsall["scenario"] = [scenstrdict[i] for i in list(egdsall["scenario"].data)]

egdfall = egdsall.to_dataframe().dropna(how="all").reset_index()
egdfall = egdfall.replace({"region":regcodes})
egdfall["region"] = egdfall["region"].loc[egdfall["region"].isin(useregs)]
egdfall = egdfall.replace({"region":reglongnamesdict})
egdfall["rcpscen"] = egdfall["scenario"].str[:6]
egdfall["luscen"] = egdfall["scenario"].str[7:]



#%% Uniform colors and shapes
centralshape = "d"
statmodel_properties = {
    'Ensemble' : ("black", centralshape,7.0), 
    # 'Ensemble' : ("black", ".",7.0), 
    'GDD + EDD' : ("red", ".",5.0), 
    'GDD + EDD + VPD' : ("orange", ".",5.0),
    'GDD + EDD + VPD + Prec' : ("purple", ".",5.0), 
    'VPD' : ("blue", ".",5.0)
}
# transpath = matplotlib.transforms.Affine2D()
# statmodel_properties = {
#     'Ensemble' : ("black", mpath.Path.unit_circle().transformed(transpath.scale(0.5))), 
#     'GDD + EDD' : ("red", mpath.Path.unit_circle().transformed(transpath.scale(0.5))), 
#     'GDD + EDD + VPD' : ("orange", mpath.Path.unit_circle().transformed(transpath.scale(0.5))),
#     'GDD + EDD + VPD + Prec' : ("purple", mpath.Path.unit_circle().transformed(transpath.scale(0.5))), 
#     'VPD' : ("blue", mpath.Path.unit_circle().transformed(transpath.scale(0.5)))
# }
statmodel_colors = {i:statmodel_properties[i][0] for i in statmodel_properties.keys()}
statmodel_shapes = {i:statmodel_properties[i][1] for i in statmodel_properties.keys()}
statmodel_sizes  = {i:statmodel_properties[i][2] for i in statmodel_properties.keys()}
#%% By region, area weights
groupvars = ["region","scenario"]
if iyield: groupvars.extend(["statmodel"])

# dumdf = dfall.groupby(groupvars).apply(lambda dfx: (dfx[contvarname] * dfx[agareavarname]).sum() / dfx[agareavarname].sum())
# dumdf = pd.DataFrame(dumdf.rename(contvarname))#.reset_index()
# dumdf = dumdf.reset_index()
dumdf = aggregate_weighted(dfall, groupvars, agareavarname)

printdf = dumdf
if iyield: printdf = printdf.query("statmodel=='Ensemble'") 
printdf.reset_index("scenario").pivot(columns="scenario")
print(printdf)

aesdict = {"y" : contvarname, "x" : "scenario"}
erraesdict = {"ymin" : "cilo","ymax" : "ciup","x" : "scenario"}
pointkwargs = {}
if iyield: 
    # aesdict.update({"color" : "statmodel", "shape" : "statmodel", "size" : "statmodel", "group" : "statmodel"})
    # aesdict.update({"color" : "statmodel", "size" : "statmodel"})
    aesdict.update({"color" : "statmodel", "shape" : "statmodel"})
    erraesdict.update({"color" : "statmodel"})
    pointkwargs.update({"size" : 5})
else:
    pointkwargs.update({"shape" : "d", "size" : 5})

posd = p9.position_dodge(0.0)
dumplot = (
    # p9.ggplot(dumdf.dropna().reset_index()) +
    # p9.geom_point(p9.aes(**aesdict), **pointkwargs) +
    p9.ggplot(dumdf.dropna().reset_index(), p9.aes(**aesdict)) +
    p9.geom_point(**pointkwargs, position = posd) +
    p9.geom_errorbar(p9.aes(**erraesdict),
        width=0.4,size=0.8, position = posd) + 
    # p9.geom_errorbar(p9.aes(ymin="cilo",ymax="ciup",x="scenario"),color="statmodel"),
    #     width=0.4,size=0.8, position = posd) + 
    p9.facet_wrap("region") +
    p9.xlab("") + 
    p9.ylab(cropstrdict[crop]+" average "+varshortnamedict[contvarname]+" change ("+deltads[contvarname].attrs["units"]+", area weighted)") +
    p9.theme_classic()
)

if iyield: 
    dumplot = (dumplot + p9.scale_color_manual(statmodel_colors) + 
        p9.scale_shape_manual(statmodel_shapes))# +
        # p9.scale_size_manual(statmodel_sizes))
    dumplot = dumplot + p9.labs(color="Yield model", shape="Yield model") 
# Rotate legends
dumplot = dumplot + p9.theme(axis_text_x = p9.element_text(angle = 45, vjust = 1.0, hjust=1))
# Add horizontal line
dumplot = dumplot + p9.geom_hline(p9.aes(yintercept=0.0), color = "gray", linetype = "dashed")
# dumplot = dumplot + p9.theme(legend_background=p9.element_blank(), legend_position=(0.28,0.27))
print(dumplot)
p9.ggsave(dumplot, varfolder + "/plot_regions_"+regtypestr+"_"+contvarname+"_areaweights")

#%% By region, NO weights
groupvars = ["region","scenario"]
if iyield: groupvars.extend(["statmodel"])

dumdf = dfall.groupby(groupvars).mean()
dumdf = pd.DataFrame(dumdf)#.reset_index()
# dumdf = dumdf.reset_index()
# dumdf = aggregate_weighted(dfall, groupvars, agareavarname)

printdf = dumdf
if iyield: printdf = printdf.query("statmodel=='Ensemble'") 
printdf.reset_index("scenario").pivot(columns="scenario")
print(printdf)

aesdict = {"y" : contvarname, "x" : "scenario"}
pointkwargs = {"size" : 5}
if iyield: 
    aesdict.update({"color" : "statmodel", "shape" : "statmodel"})
else:
    pointkwargs.update({"shape" : "d"})

dumplot = (
    p9.ggplot(dumdf.dropna().reset_index()) +
    p9.geom_point(p9.aes(**aesdict), **pointkwargs) +
    p9.geom_errorbar(p9.aes(ymin="cilo",ymax="ciup",x="scenario")) + 
    p9.facet_wrap("region") +
    p9.xlab("") + p9.ylab(cropstrdict[crop]+" average "+varshortnamedict[contvarname]+" change ("+deltads[contvarname].attrs["units"]+"no weights)") +
    p9.theme_classic()
)
if iyield: 
    dumplot = dumplot + p9.scale_color_manual(statmodel_colors) + p9.scale_shape_manual(statmodel_shapes) #+
    dumplot = dumplot + p9.labs(color="Yield model", shape="Yield model") 
# Rotate legends
dumplot = dumplot + p9.theme(axis_text_x = p9.element_text(angle = 45, vjust = 1.0, hjust=1))
# Add horizontal line
dumplot = dumplot + p9.geom_hline(p9.aes(yintercept=0.0), color = "gray", linetype = "dashed")
# dumplot = dumplot + p9.theme(legend_background=p9.element_blank(), legend_position=(0.28,0.27))
print(dumplot)
p9.ggsave(dumplot, varfolder + "/plot_regions_"+regtypestr+"_"+contvarname+"_noweights")

#%% Grand average, area weights TRANPOSED

groupvars = ["scenario"]
if iyield: groupvars.extend(["statmodel"])

dumdf = aggregate_weighted(dfall, groupvars, agareavarname)

printdf = dumdf
if iyield: printdf = printdf.query("statmodel == 'Ensemble'") 
print(printdf)

aesdict = {"y" : contvarname, "x" : "scenario"}
pointkwargs = {"size" : 5}
if iyield: 
    aesdict.update({"color" : "statmodel", "shape" : "statmodel"})
    pointkwargs.update({"size" : 10})

else:
    pointkwargs.update({"shape" : "D"})
# if iyield: 
#     aesdict.update({"color" : "statmodel", "shape" : "statmodel"})
#     pointsize = 10

dumplot = (
    p9.ggplot(dumdf.dropna().reset_index()) +
    p9.geom_point(p9.aes(**aesdict), **pointkwargs) +
    p9.geom_errorbar(p9.aes(ymin="cilo",ymax="ciup",x="scenario")) + 
    p9.ylab("") + p9.xlab(cropstrdict[crop]+" average "+varshortnamedict[contvarname]+" change ("+deltads[contvarname].attrs["units"]+", area weighted)") +
    p9.theme_classic()

)
if iyield: 
    dumplot = dumplot + p9.scale_color_manual(statmodel_colors) + p9.scale_shape_manual(statmodel_shapes) #+
    dumplot = dumplot + p9.labs(color="Yield model", shape="Yield model") 

print(dumplot)

#%% Grand average, production weights
groupvars = ["scenario"]
if iyield: groupvars.extend(["statmodel"])
dumdf = dfall.groupby(groupvars).apply(lambda dfx: (dfx[contvarname] * dfx[agprodvarname]).sum() / dfx[agprodvarname].sum())
dumdf = pd.DataFrame(dumdf.rename(contvarname))

printdf = dumdf
if iyield: printdf = printdf.query("statmodel == 'Ensemble'") 
print(printdf)

aesdict = {"x" : contvarname, "y" : "scenario"}
pointkwargs = {"size" : 5}
if iyield: 
    aesdict.update({"color" : "statmodel", "shape" : "statmodel"})
    pointkwargs.update({"size" : 10})

else:
    pointkwargs.update({"shape" : "d"})
# if iyield: 
#     aesdict.update({"color" : "statmodel", "shape" : "statmodel"})
#     pointsize = 10

dumplot = (
    p9.ggplot(dumdf.dropna().reset_index()) +
    p9.geom_point(p9.aes(**aesdict), **pointkwargs) +
    p9.ylab("") + p9.xlab(cropstrdict[crop]+" average "+varshortnamedict[contvarname]+" change ("+deltads[contvarname].attrs["units"]+", area weighted)") +
    p9.theme_classic()

)
if iyield: 
    dumplot = dumplot + p9.scale_color_manual(statmodel_colors) + p9.scale_shape_manual(statmodel_shapes) #+
    dumplot = dumplot + p9.labs(color="Yield model", shape="Yield model") 

print(dumplot)
p9.ggsave(dumplot, varfolder + "/plot_aggregated_"+contvarname+"_prodweights")
# %%
# ========================== CORRELATION PLOTS =============================
def regression_string_fit(df, yvarname, xvarname):
    model = ols(yvarname + "~" +xvarname, data = df)
    results = model.fit()
    b0str = "{0:.2g}".format(results.params["Intercept"])
    # b1str = "{0:.2g}".format(results.params[xvarname])
    b1str = ["- ", "+ "][results.params[xvarname] > 0] + "{0:.2g}".format(abs(results.params[xvarname]))
    r2str = "{0:.2g}".format(results.rsquared)
    # regstr = "R2: " + r2str + "\n" + b0str + " + " + b1str + "x" 
    regstr = "R2: " + r2str + "\n" + b0str + " " + b1str + "x" 
    return(regstr)

groupvars = ["rcpscen","region"]
# groupvars = ["scenario","region"]

# # Faceting and add grouping for yield plots
# if iyield:
#     # facetstr = "+".join(groupvars)+"~statmodel"
#     facet_function = p9.facet_grid("statmodel ~ region")
#     groupvars = groupvars + ["statmodel"]
# # else:
#     facet_function = p9.facet_grid("~".join(groupvars))

dfuse = dfall
if iyield:
    dfuse = dfall.query("statmodel=='Ensemble'")

# Regressions
dfanno = dfuse.groupby(groupvars).apply(regression_string_fit, contvarname, pftvarname).rename("equation").reset_index()
ypos = dfuse[contvarname].max()
xpos = dfuse[pftvarname].min()

dumplot = (
    p9.ggplot(dfuse.dropna(how="any",subset=[contvarname,pftvarname,"region"]),
    ) + 
    # p9.ggtitle(contvarname) + 
    p9.aes(x="PCT_PFT", y=contvarname, color="region") +
    p9.geom_point() +
    # p9.coords.coord_fixed() +
    # p9.facet_wrap("~" + "+".join(groupvars)) +
    p9.facet_grid("~".join(groupvars)) +
    # facet_function +
    p9.xlab(deflevlabelstring) +
    p9.ylab(deltads[contvarname].attrs["long_name"] + " (" + deltads[contvarname].attrs["units"] + ")") +
    p9.geom_smooth(method = "lm", se = True, show_legend = False) +
    p9.theme_classic() +
    p9.theme(legend_position = "none") +
    p9.geom_text(p9.aes(x=xpos,y=ypos,label="equation"), 
        va = "top",
        ha = "left",
        color="black", 
        data = dfanno)
)
print(dumplot)
p9.ggsave(dumplot, varfolder + "/plot_regressions_"+"_".join(groupvars) + ".png")

# %%
# ================================= CORRELATION EG PLOTS ============================
groupvars = ["rcpscen","region"]
# groupvars = ["scenario","region"]

# # Faceting and add grouping for yield plots
# if iyield:
#     # facetstr = "+".join(groupvars)+"~statmodel"
#     facet_function = p9.facet_grid("statmodel ~ region")
#     groupvars = groupvars + ["statmodel"]
# # else:
#     facet_function = p9.facet_grid("~".join(groupvars))

# dfuse = dfall
del(dfuse)
dfuse = copy.deepcopy(egdfall)
if iyield:
    dfuse = dfuse.query("statmodel=='Ensemble'")

# Regressions
dfanno = dfuse.groupby(groupvars).apply(regression_string_fit, contvarname, pftvarname).rename("equation").reset_index()
ypos = dfuse[contvarname].max()
xpos = dfuse[pftvarname].min()

dumplot = (
    p9.ggplot(dfuse.dropna(how="any",subset=[contvarname,pftvarname,"region"]),
    ) + 
    # p9.ggtitle(contvarname) + 
    p9.aes(x="PCT_PFT", y=contvarname, color="region") +
    p9.geom_point() +
    # p9.coords.coord_fixed() +
    # p9.facet_wrap("~" + "+".join(groupvars)) +
    p9.facet_grid("~".join(groupvars)) +
    # facet_function +
    p9.xlab("Diff. in "+deflevlabelstring) +
    p9.ylab(egds[contvarname].attrs["long_name"] + " (" + egds[contvarname].attrs["units"] + ")") +
    p9.geom_smooth(method = "lm", se = True, show_legend = False) +
    p9.theme_classic() +
    p9.theme(legend_position = "none") +
    p9.geom_text(p9.aes(x=xpos,y=ypos,label="equation"), 
        va = "top",
        ha = "left",
        color="black", 
        data = dfanno)
)
print(dumplot)
p9.ggsave(dumplot, varfolder + "/plot_egregressions_"+"_".join(groupvars) + ".png")


