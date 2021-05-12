# Apply statistical models (as estimated by climate-soybeans in R)
# to yearly growing-season climate output.
# FIXME: Currently only operates with a single model
# %%
from matplotlib import pyplot as plt
import importlib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import ttest_ind_from_stats

import cftime
import tqdm
import os
import glob
import re
import copy
import datetime
import numba
from numba import jit,prange
import rpy2.robjects
import sys

sys.path.append("../")
import pooled_stats 
import bootstrap_utils 
import geo_utils
# %% [markdown]
# ### Setup

# %%
# infolder = "../external/data/xavier/daily_all/"
baseinfolder = "../output_ag/dd_calc/"
baseoutfolder = "../output_ag/ag_vars/"

basersfolder = "../output_rainy_season/"
refrsfname = "../auxdata/xavier.all.rs.1981.2015.nc"

# This enables simple plots of the historical bias
iplotbias = True
if iplotbias: mskfname = "../auxdata/brmask.nc"


crops = ["single","maize"]
# crop = "single"

cropstrdict = {"single" : "Soy", "maize" : "2nd crop Maize"}

# calname = "ming595_mean"
calname = "Oct15-Mar1_Jan15-Jun15"

tempname = "temp"
tmaxname = "tmax"
tminname = "tmin"
precname = "prec"
vpdname = "vpd"

# calfolder = "fixed_calendars/"+calname+"/"
basecalfolder = baseinfolder + calname + "/"

# scens       =   next(os.walk(basecalfolder))[1]

# Default behavior is future scenarios minus historical
# Set ifutdelta to True for difference within scenarios,
# and False to enable trends (see below)
# ifutdelta   =   True
ifutdelta   =   False

# Whether to calculate future trends relative to historical
# WARNING: The implementation of trends can be quite slow, 
# so don't run this if you don't need it. Takes around 20s per variable.
# Time is not strongly dependent on nsamp (tested up to 100) since
# the bottlenecks are in the Dataframe manipulations used and
# the actual bootstrap in implemented in numba
if not ifutdelta:
    itrend      =   True
    tsyear      =   2012
    teyear      =   2050
    nsamp       =   500
    # trendvarnames = ["tempmean","tmaxmean","tminmean","precmean","vpdmean","edd","gdd","agestimate"]
    trendvarnames = ["tempmean","tmaxmean","tminmean","precmean","vpdmean","edd","gdd","agestimate","outslen","outoday","outeday"]

fscens      =   ["rcp8.5_seg", "rcp8.5_weg", "rcp8.5_cmp", "rcp2.6_seg", "rcp2.6_weg", "rcp2.6_cmp"]
hscen       =   "historical"

allscens    =   [i for i in fscens]
allscens.append(hscen)

enscodes    =   ["005","006", "008", "009"]

fsyear = 2035
feyear = 2050
hsyear = 1991
heyear = 2005

if ifutdelta:
    fsyear = 2032
    feyear = 2050
    hsyear = 2012
    heyear = 2030
    


basevalgdd = 10.0
basevaledd = 30.0    

# RDS file containing the estimated R lm() models
# modelsfname = "../../climate-soybeans/models/fixed_calendar/model.ming595_yearly_fill.xavier.single.2.rds"
modelsfname = "../../climate-soybeans/models/fixed_calendar/model.v3.ming595_yearly_fill.xavier.single.2.rds"
modelsfnamesdict = {
    "single" : "../../climate-soybeans/models/fixed_calendar/model.v3.ming595_yearly_fill.xavier.single.2.rds",
    "maize" : "../../climate-soybeans/models/fixed_calendar/model.v3.ming595_yearly_fill.xavier.maize.2.rds",

}

modelstringsdict = {"fit.fe.vpd" : "VPD",
                   "fit.fe.gddedd" : "GDD + EDD",
                   "fit.fe.gddeddvpd" : "GDD + EDD + VPD",
                   "fit.fe.gddeddvpd.prec" : "GDD + EDD + VPD + Prec"}
modelnames = list(modelstringsdict.keys())

# Dictionary translating variables as they appear in the 
# R model objects to the names in the NetCDF datasets
modelvardict = {
    "gdd1030dm" : "gdd",
    "edd30dm" : "edd",
    "vpdmeandm" : "vpdmean",
    "precmeandm" : "precmean",
    "I(precmeandm^2)" : "precmean2",
    "gdd1030" : "gdd",
    "edd30" : "edd",
    "vpdmean" : "vpdmean",
    "precmean" : "precmean",
    "I(precmean^2)" : "precmean2"
    }


metadict = {
    "tempmean" : {"long_name" : "Mean temperature", "units" : "°C"},
    "tmaxmean" : {"long_name" : "Maximum daily temperature", "units" : "°C"},
    "tminmean" : {"long_name" : "Minimum daily temperature", "units" : "°C"},
    "precmean" : {"long_name" : "Mean daily precipitation", "units" : "mm/day"},
    "vpdmean" : {"long_name" : "Mean daily vapour pressure deficit", "units" : "hPa"},
    "gdd" : {"long_name" : "Season GDD (10°C-30°C)", "units" : "°C day"},
    "edd" : {"long_name" : "Season EDD (10°C-30°C)", "units" : "°C day"},
    "agestimate" : {"long_name" : "yield, DO NOT USE, no fixed effects. Use agestimate_perc instead", "units" : "t/ha"},
    "agestimate_perc" : {"long_name" : "yield change", "units" : "%"}
}

# Output file suffix (crop + outfnamesuf)
outfnamesuf = ".allscens.estimated.nc"
if ifutdelta:
    outfnamesuf = ".futdelta.rs." + str(hsyear) + str(heyear) + "-" + str(fsyear) + str(feyear) + ".allscens.estimated.nc"

if itrend:
    trendfnamesuf = ".trend.rs." + str(tsyear) + "-" + str(teyear) +".allscens.estimated.nc"

# Also write level datasets?
iwritelevels = True
if iwritelevels:
    flevelsuf = ".levels.period." + str(fsyear) + "_" + str(feyear) + ".estimated.nc"
    hlevelsuf = ".levels.period." + str(hsyear) + "_" + str(heyear) + ".estimated.nc"

#%% Function definitions    ===========================================================================================================================================
# ====================================================================================================================================================================
# Adds metadata to several variables based on a dictionary
def add_meta_dict(ds,metadict):
    for varname in metadict.keys():
        if varname in ds.data_vars:
            ds[varname] = ds[varname].assign_attrs(metadict[varname])
    return(ds)

# Reads a R model coefficient table from an RDS file containing model fit variables
# as generated by climate-soybeans
# fname = "/media/gabriel/hd1_6tb/backup/gabriel/transicao/doutorado/climate-soybeans/models/fixed_calendar/model.ming595_yearly_fill.xavier.single.2.rds"
# fname = modelsfname
# modname = "fit.anom.gddedd"
# modname = "fit.fe.gddedd"
def read_R_model_table(fname,modname):
    if "rpy2.robjects" not in sys.modules:
        raise NameError("Remember to import rpy2.robjects")
    R = rpy2.robjects.r

    models = R.readRDS(fname)

    model = models.rx2[modname]
    if 'lm' in list(model.rclass):
        smodel = R.summary(model)
        rcoeftable = smodel.rx2["coefficients"]
        rtermnames = R.rownames(rcoeftable)
        rcolnames = R.colnames(rcoeftable)

        coefsdf = pd.DataFrame(np.asarray(rcoeftable))
        coefsdf.index = rtermnames
        coefsdf.columns = rcolnames
    elif "felm" in list(model.rclass):
        rcoefmeans = model.rx2["coefficients"]
        rcoefses = model.rx2["se"]
        coefsarr = np.vstack((np.asarray(rcoefmeans).T.squeeze(),np.asarray(rcoefses))).T

        rtermnames = R.rownames(rcoefmeans)
        colnames = ['Estimate','Std. Error']

        coefsdf = pd.DataFrame(coefsarr)
        coefsdf.index = rtermnames
        coefsdf.columns = colnames

    return(coefsdf)

# print(read_R_model_table(modelsfname,modelname))
# Reads a R model coefficient table from an RDS file containing model fit variables
# as generated by climate-soybeans and converts it to a xarray Dataset with means
# and variances
def read_R_model_dataset(modelsfname, modelname, modelvardict = None):
    coeftable = read_R_model_table(modelsfname,modelname)

    coeftable["variance"] = coeftable["Std. Error"]**2 
    coeftable = coeftable.drop(["(Intercept)", "year"], errors="ignore")[["Estimate","variance"]]
    coeftable = coeftable.T

    if modelvardict is not None:
        coeftable = coeftable.rename(modelvardict, axis = 1)

    coefdsmeans = xr.Dataset(coeftable.loc["Estimate"])
    coefdsvariances = xr.Dataset(coeftable.loc["variance"])

    coefds = pooled_stats.combine_dataset_variances_generic(coefdsmeans,coefdsvariances)
    return(coefds)


# Opens a dataset and adds scenario and member dimensions with metadata
def meta_open_dataset(fname,scen,enscode):
    ds = xr.open_dataset(fname)
    ds = ds.assign_coords({
        "year":ds.attrs["harvest_year"],
        "scenario" : scen,
        "member" : enscode
        }).expand_dims(["year","scenario","member"])
    return(ds)

# Opens a dataset concatenating all files in an ensemble member's folder,
# adding scenario and member metadata
def dataset_open_ens_folder(basecalfolder, enscode, scen, crop):
    ensfolder = basecalfolder + scen + "_" + enscode + "/"
    listensds = [meta_open_dataset(fname,scen,enscode) for fname in glob.glob(ensfolder + "/" + crop + ".*.nc")]
    ensds = xr.combine_nested(listensds, concat_dim=["year"], combine_attrs="override")
    ensds = ensds.sortby("year")
    return(ensds)

# Opens a dataset concatenating all files in an ensemble member's folder of calculated rainy season files,
# adding scenario and member metadata
def dataset_rs_open_ens_folder(basersfolder, enscode, scen, refarr):
    ensfolder = basersfolder + scen + "_" + enscode + "/"
    # listensds = [meta_open_dataset(fname,scen,enscode) for fname in glob.glob(ensfolder + "/" ".*.nc")]
    # ensds = xr.combine_nested(listensds, concat_dim=["year"], combine_attrs="override")
    ensds = xr.open_dataset(glob.glob(ensfolder + "/*.nc")[0]) # Assume there`s a single file there
    ensds = ensds.rename({"time" : "year"})
    ensds["year"] = ensds["year"].astype("int64")
    ensds = ensds.assign_coords({
        "scenario" : scen,
        "member" : enscode
        }).expand_dims(["scenario","member"])    
    ensds = ensds.sortby("year")

    ensds = ensds.interp(lat=refarr["lat"], lon=refarr["lon"], method="nearest")
    # ensds=ensds.interp_like(refarr, method="nearest")

    # Filter out in all variables points where length was <0
    ensds = ensds.where(ensds["outslen"] >0)
    return(ensds)

# Adds a suffix to all variables in a Dataset
def add_suf(ds,suf):
    return(ds.rename({i:(i + suf) for i in ds.data_vars}))

# Removes a suffix from all variables in a Dataset
def rem_suf(ds,suf):
    return(ds.rename({i:(re.sub("(.*)"+suf+"$","\g<1>",i)) for i in ds.data_vars}))

# Calculates mean and variances of all variables in a Dataset, 
# appending "_var" to the variance's variable names
def calc_ds_mean_and_var(allinds, dims = ["year","member"]):
    # Means and variances in separate Datasets
    meaninds = allinds.mean(keep_attrs = True, dim = dims)
    varinds = allinds.var(keep_attrs = True, ddof = 1, dim = dims)
    varinds = add_suf(varinds, "_var")

    # Combine and add number of observations
    alloutds = xr.merge([meaninds,varinds])
    alloutds.attrs['nobs_var'] = np.prod([len(allinds[d]) for d in dims])
    return(alloutds)

#Calculate GDD and EDD as new variables in a Dataset that has tempgdds
def calc_gdd_edd(inds,basevalgdd,basevaledd):
    inds["edd"] = inds["tempgdds"].sel(tmp = basevaledd)
    inds["gdd"] = inds["tempgdds"].sel(tmp = basevalgdd) - inds["tempgdds"].sel(tmp = basevaledd)

    inds["edd"].attrs["basevaledd"] = basevaledd
    inds["gdd"].attrs["basevalgdd"] = basevalgdd
    return(inds)

# Converts specified units from a Dataarray, using an optional exponent for higher moments (e.g. variances)
# We're not enforcing unit names here, just converting select ones
# For example, Pa/s and Pa s-1 won't be made the same
def convert_units(da, exponent = 1):
    if da.attrs['units'] == "kg m-2 s-1":
        # Convert from kg m-2 s-1 to m/s
        # da.values = da.values/1000
        da.values = da.values*(0.001**exponent)
        da.attrs['units'] = "m/s"
    else:
        da = da
    return(da)

# Applies a single model, ignoring variances and assuming all derived model
# terms are already calculated (e.g. squared variable) and have 
# properly named variables
# FIXME: Figure out a good way to calculate estimate variances
def apply_model_novar(ds,mds):
    (dsmeans, dsvariances) = pooled_stats.split_dataset_variances_generic(ds)

    (mdsmeans, mdsvariances) = pooled_stats.split_dataset_variances_generic(mds)


    # DataArray with estimated means
    estdameans = pooled_stats.sum_all_variables(dsmeans*mdsmeans)

    # Exponentiate since models are log
    estdameans = (100.0*(np.exp(estdameans)-1))
    estdameans.name = "agestimate"

    return(estdameans)

# Applies a single model on levels (i.e. not differences),
# ignoring variances and assuming all derived model
# terms are already calculated (e.g. squared variable) and have 
# properly named variables
# Keep in mind that, since we are ignoring the fixed effects here,
# levels may be only compared to each other (in percent form i.e. (Y2-Y1)/Y1), and will be in log form
# FIXME: Figure out a good way to calculate estimate variances
def apply_model_level_novar(ds,mds):
    (dsmeans, dsvariances) = pooled_stats.split_dataset_variances_generic(ds)

    (mdsmeans, mdsvariances) = pooled_stats.split_dataset_variances_generic(mds)


    # DataArray with estimated means
    estdameans = pooled_stats.sum_all_variables(dsmeans*mdsmeans)

    # Exponentiate since models are log
    # We can consider that Y = exp(B1*X) here, ignoring B0 or fixed effects
    # if we will only compare them with each other in percent such as (Y2-Y1)/Y1
    # estdameans = (100.0*(np.exp(estdameans)-1))
    estdameans = np.exp(estdameans)
    estdameans.name = "agestimate"
    estdameans = estdameans.assign_coords({"statmodel" : mds.attrs["modelstring"]}).expand_dims("statmodel")


    return(estdameans)


# Calculates the difference between all variables in a dataset, additionally calculating
# percent differences in for variables in percnames
# and also returns a _pval variable for each input variable with the p-values of a
# t-test on the difference of two means given a _var variable for each one
def calc_diff_ttest_withperc(dsboth1, dsboth2, nobs, percnames):

    (dsmeans1, dsvariances1) = pooled_stats.split_dataset_variances_generic(dsboth1)
    (dsmeans2, dsvariances2) = pooled_stats.split_dataset_variances_generic(dsboth2)

    diff = dsmeans1 - dsmeans2

    diffperc = 100.0*(dsmeans1[percnames] - dsmeans2[percnames])/dsmeans2[percnames]
    diffperc = diffperc.rename({i:(i + "_perc") for i in diffperc.data_vars})

    # t-test
    # testvarnames = selvars

    dsttest = xr.apply_ufunc(
            ttest_ind_from_stats,
            dsmeans1,
            dsvariances1**0.5,
            nobs,
            dsmeans2,
            dsvariances2**0.5,
            nobs,
            True,
            input_core_dims=[[], [], [], [], [], [], []],
            output_core_dims=[[], []],
            vectorize=True,
            # keep_attrs=True,
            dask='parallelized',
        )[1]

    # Associate the p-values of differences to the percent differences as well
    dsttest = dsttest.merge(pooled_stats.add_suf(dsttest[percnames],"_perc"))

    # Add _pval to the names
    dsttest = dsttest.rename({i:(i + "_pval") for i in dsttest.data_vars})
    
    # Variance of the difference
    dsvariances = dsvariances1 + dsvariances2
    dsvariances = dsvariances.rename({i:(i + "_var") for i in dsvariances.data_vars})
    
    diff = diff.merge(dsttest)
    diff = diff.merge(dsvariances)
    diff = diff.merge(diffperc)
    # diff.expand_dims("lev")
    # diff["lev"] = np.array(uselev)
    # (diff,dump) = xr.broadcast(diff, dsboth1)
    return(diff)

# Combines means and variances of member in a Dataset
# ndeg is the number of degrees of freedom used to calculate 
# those statistics, without the (-1) correction
# So, if those are directly observed variables on time, ndeg = nyears
# But for trends we must discount the 2 parameters estimated
def ensemble_stats_from_stats(inds, ndeg):
    # Split the dataset
    (dsmeans, dsvars) = pooled_stats.split_dataset_variances_generic(inds)

    # Aggregate means by taking a simple mean
    dsmean = dsmeans.mean(dim = "member", keep_attrs = True)

    # Aggregate variances by applying pool_variances_dask
    dsvariance = xr.apply_ufunc(pooled_stats.pool_variances_dask, dsmeans, dsvars, ndeg, \
        dask = "parallelized", input_core_dims=[["member"], ["member"], [] ],
        keep_attrs=True)

    # Join variance variables into dsmean
    dsvariance = dsvariance.rename({i:(i + "_var") for i in dsvariance.data_vars})
    dsmean = dsmean.merge(dsvariance)

    return(dsmean)

# Opens a reference RS file, fixes metadata and interpolates grid based on another DataArray/Dataset
def open_reference_rs_interp(refrsfname, gridref):
    inrefrs = xr.open_dataset(refrsfname)
    inrefcoordnames = geo_utils.get_lat_lon_name(inrefrs)
    gridrefcoordnames = geo_utils.get_lat_lon_name(gridref)
    renamedict = {i[0]:i[1] for i in zip(inrefcoordnames,gridrefcoordnames)}
    renamedict["time"] = "year"
    yearsvec = copy.deepcopy(inrefrs["time.year"].values)
    refrs = inrefrs.rename_dims(renamedict).rename_vars(renamedict)

    refrs = refrs.drop_vars("year").assign_coords({"year":yearsvec})

    # Longitudes have to be flipped to 0-360
    refrs = geo_utils.brazil_lon_flip(refrs).interp(lat=gridref["lat"], lon=gridref["lon"], method="linear")

    # Filter out in all variables points where length was <0
    refrs = refrs.where(refrs["outslen"] >0)

    # Add metadata
    refrs["outslen"].attrs["long_name"] = "Rainy season length"
    refrs["outoday"].attrs["long_name"] = "Rainy season onset DOY"
    refrs["outeday"].attrs["long_name"] = "Rainy season end DOY"
    for vname in ["outslen", "outoday", "outeday"]:
        refrs[vname].attrs["units"] = "days"
    return(refrs)

def copy_da_attrs(din,dref):
    dout = din
    for vname in din.data_vars:
        if vname in dref.data_vars:
            dout[vname].attrs = dref[vname].attrs
    return(dout)


#%%  MAIN SCRIPT ===========================================================================================================================================
# ====================================================================================================================================================================

# TODO: We must calculate anomalies of variables instead of levels first,
# before the deltas.
# Those anomalies should be in relation to the historical period mean,
# regardless if we are making future deltas.
# If we are calculating Fut-Hist, the average anomaly on the historical
# period will be 0, but the variance won't
# This means that our Yield variable of a given year (in the future or not)
# will be how much % more than the average yield of the historical yields.
# This has the major consequence of making the deltas be difference in percentages (p.p.),
# not actual percent differences.
# This will require having a historical reference period even when calculating
# deltas between future periods, so maybe calling P1 and P2 "hist" and "fut" like 
# it is now will be confusing. 

# crop = crops[0]
for crop in crops:
    print(crop)
    cropstr = cropstrdict[crop]
    modelsfname = modelsfnamesdict[crop]

    # ====================== Creating output folder
    # Output folder for that calendar
    outfolder = baseoutfolder + "/" + calname + "/" 

    # Create output folder
    os.makedirs(outfolder, exist_ok=True)

    # ====================== Opening climate scenarios
    print("Opening climate scenarios...")
    # Open all future and historical data in separate big datasets
    # listallds = [[dataset_open_ens_folder(basecalfolder, enscode, scen, crop) for scen in allscens] for enscode in enscodes]
    flistallds = [[dataset_open_ens_folder(basecalfolder, enscode, scen, crop) for scen in fscens] for enscode in enscodes]
    fbiginds = xr.combine_nested(flistallds, concat_dim=["member","scenario"])

    # Read rainy season variables
    flistrsds = [[dataset_rs_open_ens_folder(basersfolder, enscode, scen, fbiginds) for scen in fscens] for enscode in enscodes]
    frsds = xr.combine_nested(flistrsds, concat_dim=["member","scenario"])
    # Merge into the original dataset
    fbiginds = fbiginds.merge(frsds)

    # List of RS variable names
    rsvarnames = list(frsds.data_vars)

    if ifutdelta:
        hbiginds = fbiginds.copy()
    else:
        hlistallds = [[dataset_open_ens_folder(basecalfolder, enscode, scen, crop) for scen in [hscen]] for enscode in enscodes]
        hbiginds = xr.combine_nested(hlistallds, concat_dim=["member","scenario"])
        # Read rainy season variables
        hlistrsds = [[dataset_rs_open_ens_folder(basersfolder, enscode, scen, fbiginds) for scen in [hscen]] for enscode in enscodes]
        hrsds = xr.combine_nested(hlistrsds, concat_dim=["member","scenario"])
        # Merge into the original dataset
        hbiginds = hbiginds.merge(hrsds)

    # Open an observed RS reference dataset and calculate bias of the historical period
    refrs = open_reference_rs_interp(refrsfname, fbiginds).sel(year=slice(hsyear,heyear)).mean("year", keep_attrs=True)
    hmeanrs = hbiginds[rsvarnames].mean(["scenario","member","year"])
    rsbias = hmeanrs-refrs
    if iplotbias:
        inmsk = xr.open_dataarray(mskfname).isel(time=0).drop_vars("time").interp_like(fbiginds, method="nearest")
        pltbias = rsbias.where(inmsk)
        pltbias = copy_da_attrs(pltbias,refrs)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.axis("off")
        ax2.axis("off")
        ax3.axis("off")
        pltbias["outslen"].plot(ax = ax1, vmin=-40,vmax=40, levels=9, cmap="RdBu")
        pltbias["outoday"].plot(ax = ax2, vmin=-40,vmax=40, levels=9, cmap="RdBu_r")
        pltbias["outeday"].plot(ax = ax3, vmin=-40,vmax=40, levels=9, cmap="RdBu")

        fig.set_size_inches(18,4)
        fig.savefig("bias_plot.png", format="png")
    # rsbias["outoday"].plot()

    # If we are calculating trends, keep a different subset of fbiginds
    if itrend:
        tbiginds = fbiginds.sel(scenario = fscens, year = slice(tsyear,teyear))

    # Selecting years    
    fbiginds = fbiginds.sel(scenario = fscens, year = slice(fsyear,feyear))

    if ifutdelta:
        hbiginds = hbiginds.sel(scenario = fscens,  year = slice(hsyear,heyear))
    else:
        hbiginds = hbiginds.sel(scenario = hscen,  year = slice(hsyear,heyear))

    # Calculating squared precmean
    hbiginds["precmean2"] = hbiginds["precmean"]**2
    fbiginds["precmean2"] = fbiginds["precmean"]**2
    if itrend:
        tbiginds["precmean2"] = tbiginds["precmean"]**2

    # Calculate GDD and EDD (levels) and drop the tempgdds variable
    hbiginds = calc_gdd_edd(hbiginds, basevalgdd, basevaledd)
    fbiginds = calc_gdd_edd(fbiginds, basevalgdd, basevaledd)
    hbiginds = hbiginds.drop("tempgdds")
    fbiginds = fbiginds.drop("tempgdds")
    if itrend:
        tbiginds = calc_gdd_edd(tbiginds, basevalgdd, basevaledd)
        tbiginds = tbiginds.drop("tempgdds")

    # ======================= Applying stat models
    hbigestds = hbiginds
    fbigestds = fbiginds
    if itrend:
        tbigestds = tbiginds

    for modelname in modelnames:
        modelstring = modelstringsdict[modelname]    

        modelds = read_R_model_dataset(modelsfname, modelname, modelvardict)
        modelds.attrs["modelstring"] = modelstring

        # Apply model and get levels
        hbigestds = hbigestds.merge(apply_model_level_novar(hbiginds, modelds))
        fbigestds = fbigestds.merge(apply_model_level_novar(fbiginds, modelds))
        if itrend:
            tbigestds = tbigestds.merge(apply_model_level_novar(tbiginds, modelds))

    # Ensemble statmodel. Means, medians shouldn't be calculated before aggregating,
    # TODO: Medians in plot_ag_vars_trend
    hbigestds = xr.merge([hbigestds,\
        hbigestds["agestimate"].mean("statmodel").assign_coords({"statmodel" : "Ensemble"}).expand_dims("statmodel")])
    fbigestds = xr.merge([fbigestds,\
        fbigestds["agestimate"].mean("statmodel").assign_coords({"statmodel" : "Ensemble"}).expand_dims("statmodel")])
    if itrend:
            tbigestds = xr.merge([tbigestds,\
        tbigestds["agestimate"].mean("statmodel").assign_coords({"statmodel" : "Ensemble"}).expand_dims("statmodel")])

    if iwritelevels:
        # ======================= Writing level output
        hbigestds.to_netcdf(outfolder + crop + ".rs." + hlevelsuf)
        fbigestds.to_netcdf(outfolder + crop + ".rs." + flevelsuf)

    # ====================== Trend calculation
    if itrend:
        print("Applying trends, this can take a while...")
        trends = bootstrap_utils.ds_apply_reg_bootstrap(tbigestds[trendvarnames], "year", nsamp = nsamp)

        # Get historical average agestimate to calculate percentages
        baseestimate = hbigestds["agestimate"].drop("scenario").mean("year")
        trends["agestimate_perc"] = trends["agestimate"]*(100.0/baseestimate)
        trends["agestimate_perc_var"] = trends["agestimate_var"]*(100.0/baseestimate)**2

        # Aggregate ensemble member statistics
        nyears = (teyear - tsyear + 1)
        ndeg = nyears - 2 # Trend parameters estimated
        trends = ensemble_stats_from_stats(trends, ndeg)
        
        # Add the reference used
        trends = trends.merge(add_suf(refrs,"_ref"))
        trends = trends.merge(add_suf(rsbias,"_bias"))
        
        trends.attrs["nyears"] = (teyear - tsyear + 1)
        trends.attrs["bootstrap_nsamp"] = nsamp
        trends.attrs["perc_base_syear"] = hsyear
        trends.attrs["perc_base_eyear"] = heyear
        trends["agestimate_perc"].attrs["perc_base_syear"] = hsyear
        trends["agestimate_perc"].attrs["perc_base_eyear"] = heyear
        
        trends["agestimate"].attrs["long_name"] = metadict["agestimate"]["long_name"]
        trends["agestimate_perc"].attrs["long_name"] = cropstr + " yield trend"

        trends["agestimate_perc"].attrs["units"] = "% year-1"

        # Write trends output
        trends.to_netcdf(outfolder + crop + ".rs." + trendfnamesuf)

    # ======================= Calculate means and variances
    hmvds = calc_ds_mean_and_var(hbigestds, dims=["year","member"])
    fmvds = calc_ds_mean_and_var(fbigestds, dims=["year","member"])

    # ======================= t-tests
    print("Performing t-tests...")
    # Differences compared to historical, with t-test pvalues 
    # and percent differences for variables listed in the last argument
    deltattests = calc_diff_ttest_withperc(fmvds, hmvds, hmvds.attrs["nobs_var"], ["agestimate", "gdd", "edd", "vpdmean", "precmean"])


    # # Create bias corrected RS estimates, copying p-values from the originals
    # # Won't work here because we are multiplying by nyearstrend only on the plotting phase
    # deltattests = deltattests.merge(add_suf(deltattests[rsvarnames] + refrs,"bc"))
    # deltattests = deltattests.merge(add_suf(rem_suf(deltattests[[i+"_pval" for i in rsvarnames]],"_pval"),"bc_pval"))

    # Adding some metadata
    deltattests = add_meta_dict(deltattests, metadict)
    deltattests["agestimate"].attrs["long_name"] = cropstr + " " + deltattests["agestimate"].attrs["long_name"]
    deltattests["agestimate_perc"].attrs["long_name"] = cropstr + " " + deltattests["agestimate_perc"].attrs["long_name"]

    # Main output file name
    # outfname = outfolder + crop + ".allscens.estimated.nc"
    outfname = outfolder + crop + ".rs." + outfnamesuf
     
    print("Writing output file...")
    # Write output
    deltattests.to_netcdf(outfname)

#%%