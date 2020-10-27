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

minlat = -61
maxlat = 15
minlon = 273
maxlon = 345

usemons = [9,10,11,12]
uselev = 85000

plotfname = "../output_plots/deltahist_wind_" + contvarname + "_" + str(int(uselev/100))
wks_type = "png"

# Variable and level dependent plotting parameters
windrefmag = 10.0
deltarefmag = 10.0

# %%
# Set up a bounding box right outside the plotting area so the wind vectors are nice
rminlat = minlat-3
rmaxlat = maxlat+3
rminlon = minlon-3
rmaxlon = maxlon+3

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

    return(res)

def add_wind_common_resources(reswind):
    reswind.vcFillArrowsOn          = True
    reswind.vcLineArrowThicknessF   = 3.0
    reswind.vcLineArrowHeadMaxSizeF = 0.05 # Default: 0.05 (LineArrow), 0.012 (CurlyVector) 
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

contres.cnFillPalette           =   "matlab_jet"
contres.cnLevelSelectionMode    =   "ExplicitLevels"
contres.cnLevels    =   np.arange(1300,1600,25)

# Resources for reference wind plots
reswind = set_common_resources()
reswind = add_wind_common_resources(reswind)

# reswind.vcFillArrowsOn          = True
# reswind.vcLineArrowThicknessF   = 3.0
# reswind.vcLineArrowHeadMaxSizeF = 0.05 # Default: 0.05 (LineArrow), 0.012 (CurlyVector) 
reswind.vcRefMagnitudeF         = windrefmag             # define vector ref mag
# reswind.vcRefLengthF            = 0.04             # define length of vec ref
# reswind.vcMinFracLengthF        = 0.8
# reswind.vcMinDistanceF          = 0.035
# reswind.vcRefAnnoOrthogonalPosF = -0.20
# reswind.vcRefAnnoFontHeightF    = 0.005
# reswind.vcRefAnnoOn    = False

# reswind.vcGlyphStyle = "CurlyVector"
# # reswind.vcGlyphStyle = "WindBarb"



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
    figstrs.append("historical | " + str(usemon))

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

# Ngl.polyline_ndc(wks, [0,1], [0,1])
# Ngl.polyline_ndc(wks, [1,0], [0,1])
# Ngl.polyline_ndc(wks, [0,1], [0.7,0.7])
# Ngl.polyline_ndc(wks, [0,1], [0.8,0.8])
# Ngl.polyline_ndc(wks, [0,1], [0.9,0.9])
# Ngl.polyline_ndc(wks, [0,1], [0.05,0.05])

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

scontres.cnFillPalette           =   "BlueWhiteOrangeRed"
scontres.cnLevelSelectionMode    =   "ExplicitLevels"
scontres.cnLevels    =   np.arange(-50,50,10)

# Resources for reference wind plots
sreswind = set_common_resources()
sreswind = add_wind_common_resources(sreswind)

# sreswind.vcFillArrowsOn          = True
# sreswind.vcLineArrowThicknessF   = 3.0
# sreswind.vcLineArrowHeadMaxSizeF = 0.05 # Default: 0.05 (LineArrow), 0.012 (CurlyVector) 
sreswind.vcRefMagnitudeF         = deltarefmag             # define vector ref mag
# sreswind.vcRefLengthF            = 0.04             # define length of vec ref
# sreswind.vcMinFracLengthF        = 0.8
# sreswind.vcMinDistanceF          = 0.035
# sreswind.vcRefAnnoOrthogonalPosF = -0.20
# sreswind.vcRefAnnoFontHeightF    = 0.005
# sreswind.vcRefAnnoOn    = False

# sreswind.vcGlyphStyle = "CurlyVector"
# # sreswind.vcGlyphStyle = "WindBarb" 
# # sreswind.vcGlyphStyle = "FillArrow"

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
        sfigstrs.append(str(scen) + " | " + str(usemon))

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

Ngl.panel(wks,splots,[len(scenarios),len(usemons)],spanelres)



Ngl.delete_wks(wks)

# %%
