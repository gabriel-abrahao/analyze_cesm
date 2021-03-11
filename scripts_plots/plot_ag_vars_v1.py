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
import plotnine as p9
# %matplotlib inline

#%%
# REMEMBER PATHS ARE RELATIVE TO THE SCRIPT FOLDER!
motherfolder = "../"
ifutdelta = True

crop = "single"
calname = "Oct15-Mar1_Jan15-Jun15"

baseinfolder = motherfolder + "output_ag/ag_vars/" + calname + "/" # Below here should be scenario folders, and below them the files
infsuf = ".allscens.estimated.nc"
if ifutdelta:
    infsuf = ".futdelta.20122030-20322050.allscens.estimated.nc"

# Plot deforestation X effects?
idefplots = False
if idefplots:
    definfolder = motherfolder + "input/out_surfdata_fix/" 
    defpyear   = 2050
    defrefyear = 2005
    
    # These should match the ones in the CESM run files. 
    defscenfnames = {'rcp2.6_seg' : definfolder + "surfdata.pftdyn_0.9x1.25_rcp2.6_simyr1850-2100_rochedo_seg_c100323.nc",\
                     'rcp2.6_weg' : definfolder + "surfdata.pftdyn_0.9x1.25_rcp2.6_simyr1850-2100_rochedo_weg_c100323.nc",\
                     'rcp8.5_seg' : definfolder + "surfdata.pftdyn_0.9x1.25_rcp8.5_simyr1850-2100_rochedo_seg_c100319.nc",\
                     'rcp8.5_weg' : definfolder + "surfdata.pftdyn_0.9x1.25_rcp8.5_simyr1850-2100_rochedo_weg_c100319.nc"}

    defscennames = list(defscenfnames.keys())


    pftvarname = "PCT_PFT"

    # Indexes of PFTs considered as deforestation
    # FIXME: The actual Rochedo to CESM PFT calculation is more complex,
    # we have natural C4 grasses for example. This setup is only good for deltas
    defpfts = [14, 15, 16]


# Read regions? We have to if we want to make all deforestation plots
ireadregions = True
regfname =      "../regions/states_grid.nc"
regcodesfname = "../regions/states_codes.csv"
useregs = ["AM","PA","MT","MA","TO","PI"]

# Contour variable
# contvarname = "tempmean"
contvarname = "agestimate"

# Select this model TODO: change the structure to support multi-model plots
usemodelstr = "GDD+EDD"

# String wih the type of overlay, or "none" for no overlay
overlaytype = "none"
# overlaytype = "wind"

# List of strings with where we should set non-significant points to NaN
# "over" means the overlay
# sigmodes = ["over","cont"]
sigmodes = []

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

futstr = ""
if ifutdelta:
    futstr = "fut_"
plotfname =     "../output_plots/ag_new/" + "/" + contvarname + "/" + futstr + "deltahist_" + domain + "_" + contvarname + "_sig_" + str((siglev*100)) + "pp" + "_".join(sigmodes) 
efplotfname =   "../output_plots/ag_new/" + "/" + contvarname + "/" + futstr + "effects_" + domain + "_" + contvarname + "_sig_" + str((siglev*100)) + "pp" + "_".join(sigmodes) 

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
elif contvarname == "agestimate":
    contlevels  = np.arange(0,10,1)
    deltalevels = np.arange(-30,30.1,5)
    eflevels = np.arange(-18,18.1,3)
    if ifutdelta:
        deltalevels = deltalevels/2.0
        eflevels = np.arange(-12,12.1,2)
    reversecolormap = False 
    reversedeltacolormap = True
    icolormapoverride   = True
    colormapoverride    = "WhiteBlue"
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

deltafname = baseinfolder + "/" + crop + infsuf
deltads = xr.open_dataset(deltafname)

# Reorder scenarios
usescens = ['rcp2.6_cmp', 'rcp2.6_seg', 'rcp2.6_weg', 'rcp8.5_cmp', 'rcp8.5_seg', 'rcp8.5_weg']
deltads = deltads.sel(scenario = usescens)


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

    #TODO: This will only be valid in Brazil, since LU is different outside of it in each RCP.
    deltadefarr = defarr.sel(scenario = "rcp8.5_weg") - defarr.sel(scenario = "rcp8.5_seg")

    # Interpolate to match exactly the CESM grid. CESM itself is apparently far more lenient on coordinate matching than xarray.
    deltadefarr = deltadefarr.interp_like(bigdsin)

    # Rename
    deltadefarr = deltadefarr.rename("DiffDef")

#%% Read regions
if ireadregions:
    regions = xr.open_dataarray(regfname).interp_like(deltads, method = "nearest").rename("region")

regcodes = pd.read_csv(regcodesfname).set_index("code").T.to_dict("list")
regcodes = {i:regcodes[i][0] for i in regcodes.keys()}

# %% ===================== BEGIN PLOTS ==============================================
slabelstring = deltads[contvarname].attrs["long_name"] + \
    " (" + deltads[contvarname].attrs["units"] + ")"

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
if reversedeltacolormap:
    scolormap = scolormap[::-1]

# scontres.cnFillPalette           =   "BlueWhiteOrangeRed"
scontres.cnFillPalette           =   scolormap
scontres.cnLevelSelectionMode    =   "ExplicitLevels"
scontres.cnLevels    =   deltalevels


# Scenario Plots
splots = []
sfigstrs = []
scenarios = deltads.scenario.values.tolist()
# for scen in scenarios:
#     for usemon in usemons:
for scen in scenarios:
    # contres.tiMainString = usemon
    dssubset = deltads.sel(scenario = scen, statmodel = usemodelstr)
    contplot = Ngl.contour_map(wks,dssubset[contvarname].to_masked_array(),scontres)

    if "cont" in sigmodes:
        dssubset[contvarname] = dssubset[contvarname].where(dssubset[contvarname+"_pval"]<=siglev,np.nan)

    # contplot = Ngl.contour_map(wks,dssubset[contvarname].to_masked_array(),efcontres)
    
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

# %% ===================== CALCULATING EFFECTS ========================================
# Subtract all EG scenarios to their CMIP reference
efds26 = deltads.sel(scenario = ["rcp2.6_seg","rcp2.6_weg"]) - deltads.sel(scenario = ["rcp2.6_cmp"]).squeeze("scenario", drop = True)
efds85 = deltads.sel(scenario = ["rcp8.5_seg","rcp8.5_weg"]) - deltads.sel(scenario = ["rcp8.5_cmp"]).squeeze("scenario", drop = True)
efds = xr.merge([efds26,efds85])

# %% ===================== EFFECTS PLOTS ==============================================
eflabelstring = deltads[contvarname].attrs["long_name"] + " relative to CMIP5"\
    " (" + deltads[contvarname].attrs["units"] + ")"
if ifutdelta:
    eflabelstring = "Difference in " + deltads[contvarname].attrs["long_name"] + " relative to CMIP5"\
        " (" + deltads[contvarname].attrs["units"] + ")"


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
    # contres.tiMainString = usemon
    dssubset = efds.sel(scenario = scen, statmodel = usemodelstr)
    contplot = Ngl.contour_map(wks,dssubset[contvarname].to_masked_array(),efcontres)

    if "cont" in sigmodes:
        dssubset[contvarname] = dssubset[contvarname].where(dssubset[contvarname+"_pval"]<=siglev,np.nan)

    # contplot = Ngl.contour_map(wks,dssubset[contvarname].to_masked_array(),efcontres)
    
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
