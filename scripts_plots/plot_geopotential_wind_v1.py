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
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
# %matplotlib inline


#%%
# Allows for folder and variable setup specific for rainy season analysis
# isrs = True
isrs = False

# REMEMBER PATHS ARE RELATIVE TO THE SCRIPT FOLDER!
motherfolder = "../"
baseinfolder = motherfolder + "output/" # Below here should be scenario folders, and below them the files
if isrs:
    baseinfolder = motherfolder + "output_rainy_season/ensemble/"
syear = 2040
eyear = 2050
refsyear = 1990
refeyear = 2005

# Number of observations in each period (nyears*nmembers)
nobs = (eyear-syear+1)*4
refnobs = (refeyear-refsyear+1)*4

# Subtract the last year, since RS considers planting years
if isrs:
    eyear -= 1
    refeyear -= 1

# Suffix appended to the scenario name
insuf = "_ensmean_pres_" + str(syear) + "_" + str(eyear) + ".nc"
reffname = "../refdata/historical/historical_0.0_heg_ensmean_pres_" + str(refsyear) + "_" + str(refeyear) + ".nc"

# If we are using the rainy season, we have to get yearly files and subset and mean later
if isrs:
    insuf = "rs_allyears_ens_2.5_2005_2049.nc"
    reffname = "../refdata/historical/rainy_season/ensemble/historical_heg/rs_allyears_ens_2.5_1990_2004.nc"


# Contour variable
# contvarname = "Z3"
contvarname = "PRECT"
# contvarname = "TREFHT"
# contvarname = "outslen"
# contvarname = "outoday"
# contvarname = "outeday"

# String wih the type of overlay, or "none" for no overlay
# overlaytype = "none"
overlaytype = "wind"

# List of strings with where we should set non-significant points to NaN
# "over" means the overlay
sigmodes = ["over","cont"]
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

# Time type
timetype = "monthly"
# timetype = "seasonal"
if isrs:
    timetype = "yearly"

if timetype == "seasonal":
    # Months (seasons) to plot
    usemons = [1,2,3,4]
elif timetype == "monthly":
    # Months to plot
    usemons = [9,10,11,12]
elif timetype == "yearly":
    usemons = [1]
else:
    print("Unkown time type")
    print(timetype)



# Level (Pa) for all plots, or just for wind if it's a surface variable
uselev = 85000
# uselev = 20000
if isrs:
    uselev = 1

plotfname   = "../output_plots/" + timetype + "/deltahist_" + overlaytype + "_" + domain + "_" + contvarname + "_sig_" + str((siglev*100)) + "pp" + "_".join(sigmodes) +  "_" + str(int(uselev/100))
efplotfname = "../output_plots/" + timetype + "/effects_" + overlaytype + "_" + domain + "_" + contvarname   + "_sig_" + str((siglev*100)) + "pp" + "_".join(sigmodes) +  "_" + str(int(uselev/100))
difefplotfname = "../output_plots/" + timetype + "/difeffects_" + overlaytype + "_" + domain + "_" + contvarname   + "_sig_" + str((siglev*100)) + "pp" + "_".join(sigmodes) +  "_" + str(int(uselev/100))

wks_type = "png"

# ================= Variable and level dependent plotting parameters
# WARNING: Below are some predefined values of many of these 
# for specific variables which will override values in this section

# Reference magnitude for wind vectors. Units should be m/s.
windrefmag = 10.0
deltarefmag = 10.0
efrefmag = 10.0

# Default colormaps are red is higher. Reverse for things like pressure/height and precipitation
reversecolormap = False 
reversedeltacolormap = True

# Override the cnFillPallete resource in the level plots 
icolormapoverride   = True
colormapoverride    = "WhiteBlue"

# Levels
contlevels  = np.arange(1300,1600,25)
deltalevels = np.arange(-50,60,10)
eflevels = np.arange(-14,15,1)

# ===================== Predefined configurations for some variables
# Z3 850hPa
if contvarname == "Z3" and uselev == 85000:
    contlevels  = np.arange(1500,1600,10)
    # deltalevels = np.arange(-50,60,10)
    deltalevels = np.arange(-20,20.1,2)
    # eflevels = np.arange(-14,15,1)
    eflevels = np.arange(-5,5.1,1)
    reversecolormap = True 
    reversedeltacolormap = True
    icolormapoverride   = False

# # Z3 200hPa
elif contvarname == "Z3" and uselev == 20000:
    contlevels  = np.arange(12000,12500,50)
    deltalevels = np.arange(-100,120,20)
    eflevels = np.arange(-36,36.1,3)
    reversecolormap = True 
    reversedeltacolormap = True
    icolormapoverride   = False

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
    minlon = 287
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
os.makedirs(os.path.dirname(efplotfname), exist_ok=True)

# Set up a bounding box right outside the plotting area so the wind vectors are nice
rminlat = minlat-3
rmaxlat = maxlat+3
rminlon = minlon-3
rmaxlon = maxlon+3

# Month numbers to strings
monstrs = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",\
    5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",\
        10:"Oct",11:"Nov",12:"Dec"}

# Conversion values from months to seasons
seasvals = [1,1,2,2,2,3,3,3,4,4,4,1]

if timetype == "seasonal":
    # Month (season) numbers to strings
    monstrs = {1:"DJF",2:"MAM",3:"JJA",4:"SON"}
elif timetype == "yearly":
    # Month (season) numbers to strings
    monstrs = {1:""}

#%% Function definitions

# Add a suffix to all variables in a Dataset
def add_suf(ds,suf):
    return(ds.rename({i:(i + suf) for i in ds.data_vars}))

# Removes a suffix from all variables in a Dataset
def rem_suf(ds,suf):
    return(ds.rename({i:(re.sub("(.*)"+suf+"$","\g<1>",i)) for i in ds.data_vars}))

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
def calc_diff_ttest(dsboth1, dsboth2, contvarname, overvars, sigmodes, uselev, nobs):
    selvars = [contvarname]
    selvars.extend(overvars)

    (dsmeans1, dsvariances1) = split_dataset_variances(dsboth1, selvars, uselev)
    (dsmeans2, dsvariances2) = split_dataset_variances(dsboth2, selvars, uselev)

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

# poi = calc_diff_ttest(bigdsin.sel(scenario = "rcp2.6_weg"), bigdsin.sel(scenario = "rcp2.6_seg"), contvarname, overvars, uselev, nobs)
# poi
# dsboth1 = bigdsin.sel(scenario = "rcp2.6_weg")
# dsboth2 = bigdsin.sel(scenario = "rcp2.6_seg")

# Summarise rainy season stats of a dataset over a given period
# TODO: Currently just the mean
def summarise_rainy_season(ds, syear, eyear):
    outds = ds.sel(time = slice(syear,eyear)).mean(dim = "time")
    return(outds)


#%%
# ============================================= BEGIN READING FILES
# Assume that there is a folder with each scenario, and right inside 
# it is a file matching insuf
scens = os.listdir(baseinfolder)
allfnames = [glob.glob(baseinfolder + i + "/" + "*" + insuf) for i in scens]
allfnames = [j for i in allfnames for j in i]

# allitems = [{"fname" : fname, "ds" : xr.open_dataset(fname)} for fname in allfnames]
allitems = [{"fname" : fname, "ds" : xr.open_dataset(fname).\
    sel(lat = slice(rminlat,rmaxlat), lon = slice(rminlon,rmaxlon))} for fname in allfnames]
for item in allitems:
    item["ds"] = item["ds"].expand_dims("scenario")
    item["ds"]["scenario"] = pd.Index([os.path.basename(os.path.split(item["fname"])[0])])

bigdsin = xr.combine_nested([i["ds"] for i in allitems], concat_dim="scenario", combine_attrs= "override")

# Reorder scenarios
usescens = ['rcp2.6_seg', 'rcp2.6_weg', 'rcp8.5_seg', 'rcp8.5_weg']
bigdsin = bigdsin.sel(scenario = usescens)

# If rainy season, subset the period of interest and 
# calculate summaries. TODO: By default we'll get the
# the mean, but we can get other summaries such as 
# e.g. fraction above a certain number of days
# if isrs:
#     bigdsin = summarise_rainy_season(bigdsin, syear, eyear)
# if isrs:
#     bigdsin = bigdsin.expand_dims(["time","lev"])

# %%
# Read reference file, and make sure metada matches
refds = xr.open_dataset(reffname).\
    sel(lat = slice(rminlat,rmaxlat), lon = slice(rminlon,rmaxlon))
if "plev" in refds.coords:
    refds = refds.rename_dims(plev = "lev").rename_vars(plev = "lev")

refds = refds.assign_coords(lat = bigdsin["lat"].data)
refds = refds.assign_coords(lon = bigdsin["lon"].data)

# if isrs:
#     refds = summarise_rainy_season(refds, refsyear, refeyear)

# %%
# Change time type if asked
if timetype == "seasonal":
    refds["month"] = seasvals
    refds = refds.groupby("month").mean(keep_attrs = True)
    bigdsin["month"] = seasvals
    bigdsin = bigdsin.groupby("month").mean(keep_attrs = True)
elif timetype == "yearly":
    refds = refds.expand_dims(dim={"month":np.array([1]), "lev":np.array([1])})
    bigdsin = bigdsin.expand_dims(dim={"month":np.array([1]), "lev":np.array([1])})
    

# .plot(col = "month", vmin = 293, cmap = "YlOrRd")
# refds["TREFHT"].plot(col = "month", col_wrap=3, vmin = 293, cmap = "YlOrRd")


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

#%%
# Delta between scenarios and reference
# deltads = bigdsin - refds
deltads = calc_diff_ttest(bigdsin, refds, contvarname, overvars, sigmodes, uselev, nobs)

# Copy each variable's metadatada
for varname in bigdsin.data_vars:
    if varname in deltads.data_vars:
        for attname in bigdsin[varname].attrs:
            deltads[varname].attrs[attname] = bigdsin[varname].attrs[attname]

# %%
# ===================== BEGIN PLOTS
labelstring = refds[contvarname].attrs["long_name"] + \
    " (" + refds[contvarname].attrs["units"] + ")"

reflabelstring = labelstring + "~C~ Wind at " + str(int(uselev/100)) + "hPa (m/s)| " + \
    str(refsyear) + "-" + str(refeyear) 

wksres = Ngl.Resources()
wksres.wkHeight = 2048
wksres.wkWidth = 2048
wks = Ngl.open_wks(wks_type,plotfname,wksres)  # Open a workstation.

# Here we coarsen the wind array so the map doesn't get cramped
# refds = refds.coarsen(lat = 2, lon = 2, boundary = "trim").mean()

# Function to set common resources
def set_common_resources():
    res = Ngl.Resources()

    res.nglDraw         =   False
    res.nglFrame         =   False

    res.sfXArray               = refds["lon"].values
    res.sfYArray               = refds["lat"].values

    res.vfXArray               = refds["lon"].values
    res.vfYArray               = refds["lat"].values

    res.mpLimitMode     = "LatLon"  # Change the area of the map
    res.mpMinLatF       =  minlat   # viewed.
    res.mpMaxLatF       =  maxlat  
    res.mpMinLonF       =  minlon
    res.mpMaxLonF       =  maxlon

    res.tmXBOn          =   False
    res.tmYLOn          =   False

    res.mpGridAndLimbOn =   False

    res.mpGeophysicalLineThicknessF = 3.0
    res.mpProvincialLineThicknessF  = 2.0
    res.mpDataSetName         = "Earth..4"
    res.mpDataBaseVersion     = "MediumRes"
    res.mpOutlineSpecifiers   = "Brazil:states"

    return(res)

def add_wind_common_resources(reswind):
    reswind.vcFillArrowsOn          = True
    reswind.vcLineArrowThicknessF   = 3.0
    reswind.vcLineArrowHeadMaxSizeF = 0.04 # Default: 0.05 (LineArrow), 0.012 (CurlyVector) 
    # reswind.vcRefMagnitudeF         = windrefmag             # define vector ref mag
    reswind.vcRefLengthF            = 0.04             # define length of vec ref
    reswind.vcMinFracLengthF        = 0.8
    reswind.vcMinDistanceF          = 0.035
    reswind.vcRefAnnoOrthogonalPosF = -0.20
    reswind.vcRefAnnoFontHeightF    = 0.005
    reswind.vcRefAnnoOn    = False

    reswind.vcGlyphStyle = "CurlyVector"

    return(reswind)

# Resources for reference contour/fill plots
contres = set_common_resources()

contres.cnFillOn               = True
contres.cnLinesOn              = False
contres.cnLineLabelsOn         = False
contres.cnFillMode             = "RasterFill"
contres.lbLabelBarOn           = False

colormap = Ngl.read_colormap_file("matlab_jet")
if reversecolormap:
    colormap = colormap[::-1]
contres.cnFillPalette           =   colormap

# Override
if icolormapoverride:
    contres.cnFillPalette           =   colormapoverride


contres.cnLevelSelectionMode    =   "ExplicitLevels"
contres.cnLevels    =   contlevels

# Resources for reference wind plots
reswind = set_common_resources()
reswind = add_wind_common_resources(reswind)

reswind.vcRefMagnitudeF         = windrefmag             # define vector ref mag

# Reference Plots
plots = []
figstrs = []
for usemon in usemons:
    dssubset = refds.sel(lev = uselev, month = usemon)
    contplot = Ngl.contour_map(wks,dssubset[contvarname].to_masked_array(),contres)

    if overlaytype == "wind":
        windplot = Ngl.vector(wks,dssubset["U"].to_masked_array(),dssubset["V"].to_masked_array(),reswind)
        Ngl.overlay(contplot,windplot)
    
    plots.append(contplot)
    figstrs.append("historical | " + monstrs[usemon])

# Ngl.overlay(contplot,windplot)
# Ngl.maximize_plot(wks, contplot)
# Ngl.draw(contplot)
# Ngl.frame(wks)
# poi = Ngl.contour_map(wks,refds["Z3"].sel(lev = uselev, month = 6).to_masked_array(),contres)

panelres                                  = Ngl.Resources()
panelres.nglPanelFigureStrings            = figstrs
panelres.nglPanelFigureStringsFontHeightF = 0.01
panelres.nglPanelLabelBar                 = True     # Turn on panel labelbar
panelres.nglPanelLabelBarLabelFontHeightF = 0.01    # Labelbar font height
panelres.nglPanelLabelBarHeightF          = 0.03   # Height of labelbar
# panelres.nglPanelLabelBarWidthF           = 0.700    # Width of labelbar
# panelres.nglPanelTop                      = 0.5
panelres.nglPanelBottom                      = 0.05
panelres.nglPanelRight                      = 0.2

panelres.lbTitleString          =   reflabelstring
panelres.lbTitlePosition          =   "Bottom"
panelres.lbTitleFontHeightF          =   0.01
panelres.lbJustification          = "TopCenter"

panelres.nglFrame = False

# Ngl.panel(wks,plots,[1,len(usemons)],panelres)
Ngl.panel(wks,plots,[len(usemons),1],panelres)

# ===============================  Resources for scenario plots ===================
slabelstring = labelstring + "~C~ Wind at " + str(int(uselev/100)) + "hPa (m/s)| " + \
    "(" + str(syear) + "-" + str(eyear) + ") - " + \
        "(" + str(refsyear) + "-" + str(refeyear) + ")"



# Resources for reference contour/fill plots
scontres = set_common_resources()

scontres.cnFillOn               = True
scontres.cnLinesOn              = False
scontres.cnLineLabelsOn         = False
scontres.cnFillMode             = "RasterFill"

scontres.lbLabelBarOn           = False

scolormap = Ngl.read_colormap_file("BlueWhiteOrangeRed")
if reversedeltacolormap:
    scolormap = scolormap[::-1]

# scontres.cnFillPalette           =   "BlueWhiteOrangeRed"
scontres.cnFillPalette           =   scolormap
scontres.cnLevelSelectionMode    =   "ExplicitLevels"
scontres.cnLevels    =   deltalevels

# Resources for reference wind plots
sreswind = set_common_resources()
sreswind = add_wind_common_resources(sreswind)

sreswind.vcRefMagnitudeF         = deltarefmag             # define vector ref mag



# Scenario Plots
splots = []
sfigstrs = []
scenarios = deltads.scenario.values.tolist()
# for scen in scenarios:
#     for usemon in usemons:
for usemon in usemons:
    for scen in scenarios:
        # contres.tiMainString = usemon
        dssubset = deltads.sel(lev = uselev, month = usemon, scenario = scen)
        contplot = Ngl.contour_map(wks,dssubset[contvarname].to_masked_array(),scontres)

        if "cont" in sigmodes:
            dssubset[contvarname] = dssubset[contvarname].where(dssubset[contvarname+"_pval"]<=siglev,np.nan)

        # contplot = Ngl.contour_map(wks,dssubset[contvarname].to_masked_array(),efcontres)
        
        if "over" in sigmodes and len(overvars) != 0:
            # Here we mask out only vectors that are non-significant in BOTH dimensions (U, V)
            overvars_pvals = [i+"_pval" for i in overvars]
            overmask = xr.apply_ufunc(np.logical_or, (dssubset[overvars_pvals[0]] <= siglev), (dssubset[overvars_pvals[1]] <= siglev))
            # dssubset = dssubset.merge(dssubset[overvars].where(rem_suf(dssubset[[i+"_pval" for i in overvars]],"_pval")<=siglev,np.nan), overwrite_vars=overvars)
            dssubset = dssubset.merge(dssubset[overvars].where(overmask,np.nan), overwrite_vars=overvars)

        if overlaytype == "wind":
            windplot = Ngl.vector(wks,dssubset["U"].to_masked_array(),dssubset["V"].to_masked_array(),reswind)
            Ngl.overlay(contplot,windplot)

        splots.append(contplot)
        sfigstrs.append(str(scen) + " | " + monstrs[usemon])

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
spanelres.nglPanelLeft                      = 0.2

# spanelres.nglMaximize = True
# spanelres.nglFrame = False

spanelres.lbTitleString          =   slabelstring
spanelres.lbTitlePosition          =   "Bottom"
spanelres.lbTitleFontHeightF          =   0.01
spanelres.lbJustification          = "TopCenter"

if addsiglabel:
    spanelres.lbTitleString = spanelres.lbTitleString + "~C~Showing differences significant at " + str(siglev*100) + "%"

Ngl.panel(wks,splots,[len(usemons),len(scenarios)],spanelres)

# print(Ngl.retrieve_colormap(wks))
# print(Ngl.read_colormap_file("BlueWhiteOrangeRed"))

Ngl.delete_wks(wks)

# %%
# Calculate effects, differences between scenarios
allefs = []
# DEF_2.6 
# allefs.append(bigdsin.sel(scenario = "rcp2.6_weg") - bigdsin.sel(scenario = "rcp2.6_seg"))
allefs.append(calc_diff_ttest(bigdsin.sel(scenario = "rcp2.6_weg"), bigdsin.sel(scenario = "rcp2.6_seg"), contvarname, overvars, sigmodes, uselev, nobs))
allefs[len(allefs)-1] = allefs[len(allefs)-1].expand_dims("scenario")
allefs[len(allefs)-1]["scenario"] = pd.Index(["DEF_2.6"])
# DEF_8.5
# allefs.append(bigdsin.sel(scenario = "rcp8.5_weg") - bigdsin.sel(scenario = "rcp8.5_seg"))
allefs.append(calc_diff_ttest(bigdsin.sel(scenario = "rcp8.5_weg"), bigdsin.sel(scenario = "rcp8.5_seg"), contvarname, overvars, sigmodes, uselev, nobs))
allefs[len(allefs)-1] = allefs[len(allefs)-1].expand_dims("scenario")
allefs[len(allefs)-1]["scenario"] = pd.Index(["DEF_8.5"])
# GHG_SEG
# allefs.append(bigdsin.sel(scenario = "rcp8.5_seg") - bigdsin.sel(scenario = "rcp2.6_seg"))
allefs.append(calc_diff_ttest(bigdsin.sel(scenario = "rcp8.5_seg"), bigdsin.sel(scenario = "rcp2.6_seg"), contvarname, overvars, sigmodes, uselev, nobs))
allefs[len(allefs)-1] = allefs[len(allefs)-1].expand_dims("scenario")
allefs[len(allefs)-1]["scenario"] = pd.Index(["GHG_SEG"])
# GHG_WEG
# allefs.append(bigdsin.sel(scenario = "rcp8.5_weg") - bigdsin.sel(scenario = "rcp2.6_weg"))
allefs.append(calc_diff_ttest(bigdsin.sel(scenario = "rcp8.5_weg"), bigdsin.sel(scenario = "rcp2.6_weg"), contvarname, overvars, sigmodes, uselev, nobs))
allefs[len(allefs)-1] = allefs[len(allefs)-1].expand_dims("scenario")
allefs[len(allefs)-1]["scenario"] = pd.Index(["GHG_WEG"])
# GHG_DEF
# allefs.append(bigdsin.sel(scenario = "rcp8.5_weg") - bigdsin.sel(scenario = "rcp2.6_seg"))
allefs.append(calc_diff_ttest(bigdsin.sel(scenario = "rcp8.5_weg"), bigdsin.sel(scenario = "rcp2.6_seg"), contvarname, overvars, sigmodes, uselev, nobs))
allefs[len(allefs)-1] = allefs[len(allefs)-1].expand_dims("scenario")
allefs[len(allefs)-1]["scenario"] = pd.Index(["GHG_DEF"])

# Combine all effects using a scenario dimension
efds = xr.combine_nested(allefs, concat_dim= "scenario")


# Copy attributes for each variable
for varname in efds.data_vars:
    if varname in bigdsin.data_vars:
        for attname in bigdsin[varname].attrs:
            efds[varname].attrs[attname] = bigdsin[varname].attrs[attname]


# %%
# ======================== BEGIN EFFECTS PLOTS
# Effects plots. We can copy over the resources from the scenario plots
# del wks
wksres = Ngl.Resources()
wksres.wkHeight = 2048
wksres.wkWidth = 2048
wks = Ngl.open_wks(wks_type,efplotfname,wksres)  # Open a workstation.

eflabelstring = labelstring + "~C~ Wind at " + str(int(uselev/100)) + "hPa (m/s)"

# Deep copying resources
efcontres = copy.deepcopy(scontres)
efreswind = copy.deepcopy(sreswind)
efpanelres = copy.deepcopy(spanelres)

# Changing some plot resources
efcontres.cnLevels    =   eflevels
efreswind.vcRefMagnitudeF = efrefmag
# del(efreswind.vcRefMagnitudeF)
del(efreswind.vcRefAnnoOrthogonalPosF)
del(efreswind.vcRefAnnoFontHeightF)
del(efreswind.vcRefAnnoOn)
efreswind.vcRefAnnoOrthogonalPosF = -1.0
efreswind.vcRefAnnoFontHeightF    = 0.03
efreswind.vcMagnitudeFormat = "*+^sg" 
efreswind.vcRefAnnoOn    = True
efreswind.vcRefAnnoString2On = False

# Effects Plots
efplots = []
effigstrs = []
scenarios = efds.scenario.values.tolist()
# scen = scenarios[0]
# usemon = usemons[0]
for usemon in usemons:
    for scen in scenarios:
        # contres.tiMainString = usemon
        dssubset = efds.sel(lev = uselev, month = usemon, scenario = scen)
        if "cont" in sigmodes:
            dssubset[contvarname] = dssubset[contvarname].where(dssubset[contvarname+"_pval"]<=siglev,np.nan)

        contplot = Ngl.contour_map(wks,dssubset[contvarname].to_masked_array(),efcontres)
        
        if "over" in sigmodes and len(overvars) != 0:
            # Here we mask out only vectors that are non-significant in BOTH dimensions (U, V)
            overvars_pvals = [i+"_pval" for i in overvars]
            overmask = xr.apply_ufunc(np.logical_or, (dssubset[overvars_pvals[0]] <= siglev), (dssubset[overvars_pvals[1]] <= siglev))
            # dssubset = dssubset.merge(dssubset[overvars].where(rem_suf(dssubset[[i+"_pval" for i in overvars]],"_pval")<=siglev,np.nan), overwrite_vars=overvars)
            dssubset = dssubset.merge(dssubset[overvars].where(overmask,np.nan), overwrite_vars=overvars)

        if overlaytype == "wind":
            windplot = Ngl.vector(wks,dssubset["U"].to_masked_array(),dssubset["V"].to_masked_array(),efreswind)
            Ngl.overlay(contplot,windplot)

        efplots.append(contplot)
        effigstrs.append(str(scen) + " | " + monstrs[usemon])

efpanelres.nglPanelFigureStrings = effigstrs
efpanelres.lbTitleString = eflabelstring

if addsiglabel:
    efpanelres.lbTitleString = efpanelres.lbTitleString + "~C~Showing differences significant at " + str(siglev*100) + "%"

efpanelres.nglPanelBottom = 0.0
efpanelres.nglPanelTop = 1.0
efpanelres.nglPanelLeft = 0.0
efpanelres.nglPanelRight = 1.0

Ngl.panel(wks,efplots,[len(usemons),len(scenarios)],efpanelres) #Freezing here

Ngl.delete_wks(wks)
print(efplotfname)
# %%
# DIFFERENCES IN DEFORESTATION EFFECTS
defdifds = calc_diff_ttest(efds.sel(scenario = "DEF_8.5"), efds.sel(scenario = "DEF_2.6"), contvarname, overvars, sigmodes, uselev, nobs)
defdifds = defdifds.expand_dims("scenario")
defdifds["scenario"] = pd.Index(["DEF_DIF"])

selscens = ["DEF_8.5", "DEF_2.6", "DEF_DIF"]
defdifds = defdifds.merge(efds).sel(scenario = selscens)
defdifds

# %%
# ======================== BEGIN DIFFERENCES IN DEFORESTATION EFFECTS PLOTS
#TODO: JUST PASTED FROM PREVIOUS CELLS, THIS DOES NOT WORK YET
# del wks
wksres = Ngl.Resources()
wksres.wkHeight = 2048
wksres.wkWidth = 2048
wks = Ngl.open_wks(wks_type,difefplotfname,wksres)  # Open a workstation.

# Effects Plots
efplots = []
effigstrs = []
scenarios = efds.scenario.values.tolist()
# scen = scenarios[0]
# usemon = usemons[0]
for usemon in usemons:
    for scen in scenarios:
        # contres.tiMainString = usemon
        dssubset = efds.sel(lev = uselev, month = usemon, scenario = scen)
        if "cont" in sigmodes:
            dssubset[contvarname] = dssubset[contvarname].where(dssubset[contvarname+"_pval"]<=siglev,np.nan)

        contplot = Ngl.contour_map(wks,dssubset[contvarname].to_masked_array(),efcontres)
        
        if "over" in sigmodes and len(overvars) != 0:
            # Here we mask out only vectors that are non-significant in BOTH dimensions (U, V)
            overvars_pvals = [i+"_pval" for i in overvars]
            overmask = xr.apply_ufunc(np.logical_or, (dssubset[overvars_pvals[0]] <= siglev), (dssubset[overvars_pvals[1]] <= siglev))
            # dssubset = dssubset.merge(dssubset[overvars].where(rem_suf(dssubset[[i+"_pval" for i in overvars]],"_pval")<=siglev,np.nan), overwrite_vars=overvars)
            dssubset = dssubset.merge(dssubset[overvars].where(overmask,np.nan), overwrite_vars=overvars)

        if overlaytype == "wind":
            windplot = Ngl.vector(wks,dssubset["U"].to_masked_array(),dssubset["V"].to_masked_array(),efreswind)
            Ngl.overlay(contplot,windplot)

        efplots.append(contplot)
        effigstrs.append(str(scen) + " | " + monstrs[usemon])

Ngl.delete_wks(wks)
print(efplotfname)