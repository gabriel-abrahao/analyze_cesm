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
import shapefile
from scipy.stats import ttest_ind_from_stats
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import plotnine as p9
# %matplotlib inline

sys.path.append("../")
import pooled_stats 
import geo_utils

#%%
# REMEMBER PATHS ARE RELATIVE TO THE SCRIPT FOLDER!
motherfolder = "../"

domain = "BR" # Zoom in Brazil

# Read agricultural observed data
agfname         = "../regions/agdata.nc"
agyear          = 2016
agsareavarname = "sharea"
agsprodvarname = "stprod"
agm2areavarname = "m2harea"
agm2prodvarname = "m2tprod"

plotfname =   "../output_plots/ag_new/" + "/" + "census" + "/" + "agvars_" + str(agyear) + "_" + domain

#%%
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

#%%
# Create the output folder
os.makedirs(os.path.dirname(plotfname), exist_ok=True)

# Set up a bounding box right outside the plotting area so the wind vectors are nice
rminlat = minlat-3
rmaxlat = maxlat+3
rminlon = minlon-3
rmaxlon = maxlon+3

#%% Read agricultural data
agdsin = xr.open_dataset(agfname).sel(lat = slice(rminlat,rmaxlat), lon = slice(rminlon,rmaxlon), year=agyear).drop("year")

agdsin["shareafrac"] = agdsin["shareafrac"]*100.0
agdsin["shareafrac"] = agdsin["shareafrac"].where(agdsin["shareafrac"] >= 0.1)
agdsin["m2hareafrac"] = agdsin["m2hareafrac"]*100.0
agdsin["m2hareafrac"] = agdsin["m2hareafrac"].where(agdsin["m2hareafrac"] >= 0.1)

#%%

wksres = Ngl.Resources()
wksres.wkHeight = 2048
wksres.wkWidth = 2048
wks = Ngl.open_wks("png",plotfname,wksres)  # Open a workstation.

# Here we coarsen the wind array so the map doesn't get cramped
# deltads = deltads.coarsen(lat = 2, lon = 2, boundary = "trim").mean()

# Function to set common resources
def set_common_resources():
    res = Ngl.Resources()

    res.nglDraw         =   False
    res.nglFrame         =   False

    res.sfXArray               = agdsin["lon"].values
    res.sfYArray               = agdsin["lat"].values

    res.vfXArray               = agdsin["lon"].values
    res.vfYArray               = agdsin["lat"].values

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
areares = set_common_resources()

areares.cnFillOn               = True
areares.cnLinesOn              = False
areares.cnLineLabelsOn         = False
areares.cnFillMode             = "RasterFill"

areares.lbLabelBarOn           = False

areacolormap = Ngl.read_colormap_file("cmocean_speed")
# areacolormap = areacolormap[::-1]

areares.cnFillPalette           =   areacolormap
areares.cnLevelSelectionMode    =   "ExplicitLevels"
areares.cnLevels    =   np.array([0,5,10,20,30,40])

areaplots = []
areafigstrs = []

areaplot = Ngl.contour_map(wks,agdsin["shareafrac"].to_masked_array(),areares)
areaplots.append(areaplot)
areafigstrs.append("Soy")

areaplot = Ngl.contour_map(wks,agdsin["m2hareafrac"].to_masked_array(),areares)
areaplots.append(areaplot)
areafigstrs.append("2nd Crop maize")

areapanelres                                  = Ngl.Resources()
areapanelres.nglPanelFigureStrings            = areafigstrs
areapanelres.nglPanelFigureStringsFontHeightF = 0.01
areapanelres.nglPanelLabelBar                 = True     # Turn on panel labelbar
# areapanelres.nglPanelLabelBar                 = True     # Turn on panel labelbar
areapanelres.nglPanelLabelBarLabelFontHeightF = 0.015    # Labelbar font height
areapanelres.nglPanelLabelBarHeightF          = 0.04   # Height of labelbar
# areapanelres.nglPanelLabelBarWidthF           = 0.700    # Width of labelbar
# areapanelres.nglPanelTop                      = 0.75
areapanelres.nglPanelBottom                      = 0.05
# areapanelres.nglPanelLeft                      = 0.2

# areapanelres.nglMaximize = True
# areapanelres.nglFrame = False

areapanelres.lbTitleString          =   "Harvested area (% of cell area)"
areapanelres.lbTitlePosition          =   "Bottom"
areapanelres.lbTitleFontHeightF          =   0.015
areapanelres.lbJustification          = "TopCenter"

Ngl.panel(wks,areaplots,[1,2],areapanelres)

# print(Ngl.retrieve_colormap(wks))
# print(Ngl.read_colormap_file("BlueWhiteOrangeRed"))

Ngl.delete_wks(wks)
