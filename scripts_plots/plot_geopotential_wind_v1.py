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
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
# %matplotlib inline

#%%
# REMEMBER PATHS ARE RELATIVE TO THE SCRIPT FOLDER!
motherfolder = "../"
baseinfolder = motherfolder + "output/" # Below here should be scenario folders, and below them the files
syear = 2040
eyear = 2050
insuf = "_ensmean_pres_" + str(syear) + "_" + str(eyear) + ".nc"
refsyear = 1990
refeyear = 2005

reffname = "../refdata/historical/historical_0.0_heg_ensmean_pres_" + str(refsyear) + "_" + str(refeyear) + ".nc"

contvarname = "Z3"

domain = "SAMATL" # South america and Atlantic Ocean
domain = "BR"

usemons = [9,10,11,12]
uselev = 85000
# uselev = 20000

plotfname = "../output_plots/deltahist_wind_" + domain + "_" + contvarname + "_" + str(int(uselev/100))
efplotfname = "../output_plots/effects_wind_" + domain + "_" + contvarname + "_" + str(int(uselev/100))
wks_type = "png"

# Variable and level dependent plotting parameters
windrefmag = 10.0
deltarefmag = 10.0
efrefmag = 10.0

# Default colormaps are red is higher. Reverse for things like pressure/height and precipitation
reversecolormap = True 

# Z3 850hPa
contlevels  = np.arange(1300,1600,25)
deltalevels = np.arange(-50,60,10)
eflevels = np.arange(-14,15,1)

# # Z3 200hPa
# contlevels  = np.arange(12000,12500,50)
# deltalevels = np.arange(-100,120,20)

# Remove the 0 in the middle of deltalevels
deltalevels = np.delete(deltalevels,np.where(deltalevels == 0))
eflevels = np.delete(eflevels,np.where(eflevels == 0))

# ======== Domains
# South America and part of the adjoining Atlantic Ocean
if domain == "SAMA":
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

# %%
# Set up a bounding box right outside the plotting area so the wind vectors are nice
rminlat = minlat-3
rmaxlat = maxlat+3
rminlon = minlon-3
rmaxlon = maxlon+3

# Month numbers to strings
monstrs = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",\
    5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",\
        10:"Oct",11:"Nov",12:"Dec"}

#%%
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
# poi = bigdsin["T"].isel(lev = 0, month = 0).sel(scenario = usescens)
# poi.mean(dim = ["lat","lon"])

# %%
# Read reference file, and make sure metada matches
refds = xr.open_dataset(reffname).\
    sel(lat = slice(rminlat,rmaxlat), lon = slice(rminlon,rmaxlon))
refds = refds.rename_dims(plev = "lev").rename_vars(plev = "lev")

refds = refds.assign_coords(lat = bigdsin["lat"].data)
refds = refds.assign_coords(lon = bigdsin["lon"].data)


# %%
# Delta between scenarios and reference
deltads = bigdsin - refds

# %%
# ===================== BEGIN PLOTS
labelstring = refds[contvarname].attrs["long_name"] + \
    " (" + refds[contvarname].attrs["units"] + ")"

reflabelstring = labelstring + "~C~ @" + str(int(uselev/100)) + "hPa | " + \
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

contres.cnLevelSelectionMode    =   "ExplicitLevels"
contres.cnLevels    =   contlevels

# Resources for reference wind plots
reswind = set_common_resources()
reswind = add_wind_common_resources(reswind)

reswind.vcRefMagnitudeF         = windrefmag             # define vector ref mag

# Plots
plots = []
figstrs = []
for usemon in usemons:
    # contres.tiMainString = usemon
    contplot = Ngl.contour_map(wks,refds["Z3"].sel(lev = uselev, month = usemon).to_masked_array(),contres)
    dswind = refds.sel(lev = uselev, month = usemon)
    windplot = Ngl.vector(wks,dswind["U"].to_masked_array(),dswind["V"].to_masked_array(),reswind)
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
slabelstring = labelstring + "~C~ @" + str(int(uselev/100)) + "hPa | " + \
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
if reversecolormap:
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
        contres.tiMainString = usemon
        contplot = Ngl.contour_map(wks,deltads[contvarname].sel(lev = uselev, month = usemon, scenario = scen).to_masked_array(),scontres)
        dswind = deltads.sel(lev = uselev, month = usemon, scenario = scen)
        windplot = Ngl.vector(wks,dswind["U"].to_masked_array(),dswind["V"].to_masked_array(),sreswind)
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

Ngl.panel(wks,splots,[len(usemons),len(scenarios)],spanelres)

# print(Ngl.retrieve_colormap(wks))
# print(Ngl.read_colormap_file("BlueWhiteOrangeRed"))

Ngl.delete_wks(wks)

# %%
# Calculate effects, differences between scenarios
allefs = []
# DEF_2.6 
allefs.append(bigdsin.sel(scenario = "rcp2.6_weg") - bigdsin.sel(scenario = "rcp2.6_seg"))
allefs[0] = allefs[0].expand_dims("scenario")
allefs[0]["scenario"] = pd.Index(["DEF_2.6"])
# DEF_8.5
allefs.append(bigdsin.sel(scenario = "rcp8.5_weg") - bigdsin.sel(scenario = "rcp8.5_seg"))
allefs[1] = allefs[1].expand_dims("scenario")
allefs[1]["scenario"] = pd.Index(["DEF_8.5"])
# GHG_SEG
allefs.append(bigdsin.sel(scenario = "rcp8.5_seg") - bigdsin.sel(scenario = "rcp2.6_seg"))
allefs[2] = allefs[2].expand_dims("scenario")
allefs[2]["scenario"] = pd.Index(["GHG_SEG"])
# GHG_WEG
allefs.append(bigdsin.sel(scenario = "rcp8.5_weg") - bigdsin.sel(scenario = "rcp2.6_weg"))
allefs[3] = allefs[3].expand_dims("scenario")
allefs[3]["scenario"] = pd.Index(["GHG_WEG"])
# GHG_DEF
allefs.append(bigdsin.sel(scenario = "rcp8.5_weg") - bigdsin.sel(scenario = "rcp2.6_seg"))
allefs[4] = allefs[4].expand_dims("scenario")
allefs[4]["scenario"] = pd.Index(["GHG_DEF"])

efds = xr.combine_nested(allefs, concat_dim= "scenario")

# %%
# Effects plots. We can copy over the resources from the scenario plots
# del wks
wksres = Ngl.Resources()
wksres.wkHeight = 2048
wksres.wkWidth = 2048
wks = Ngl.open_wks(wks_type,efplotfname,wksres)  # Open a workstation.

eflabelstring = labelstring + "~C~ @" + str(int(uselev/100)) + "hPa"

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
# for scen in scenarios:
#     for usemon in usemons:
for usemon in usemons:
    for scen in scenarios:
        contres.tiMainString = usemon
        contplot = Ngl.contour_map(wks,efds[contvarname].sel(lev = uselev, month = usemon, scenario = scen).to_masked_array(),efcontres)
        dswind = efds.sel(lev = uselev, month = usemon, scenario = scen)
        windplot = Ngl.vector(wks,dswind["U"].to_masked_array(),dswind["V"].to_masked_array(),efreswind)
        Ngl.overlay(contplot,windplot)
        efplots.append(contplot)
        effigstrs.append(str(scen) + " | " + monstrs[usemon])

efpanelres.nglPanelFigureStrings = effigstrs
efpanelres.lbTitleString = eflabelstring

efpanelres.nglPanelBottom = 0.0
efpanelres.nglPanelTop = 1.0
efpanelres.nglPanelLeft = 0.0
efpanelres.nglPanelRight = 1.0

Ngl.panel(wks,efplots,[len(usemons),len(scenarios)],efpanelres) #Freezing here

Ngl.delete_wks(wks)
# %%
