# Plot deforestation scenarios
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
import plotnine as p9
# %matplotlib inline

#%%
# REMEMBER PATHS ARE RELATIVE TO THE SCRIPT FOLDER!
motherfolder = "../"
infolder = motherfolder + "input/out_surfdata_fix/" 
pyear = 2050
# insuf = "_ensmean_pres_" + str(syear) + "_" + str(eyear) + ".nc"

# Reference year and scenarios. All RCP and Rochedo scenarios should be equal in 2005
refyear = 2005
refscen = "rcp2.6"

# reffname = "../refdata/historical/historical_0.0_heg_ensmean_pres_" + str(refsyear) + "_" + str(refeyear) + ".nc"

# contvarname = "Z3"
# contvarname = "PRECT"
pftvarname = "PCT_PFT"

# Indexes of PFTs considered as deforestation
# FIXME: The actual Rochedo to CESM PFT calculation is more complex,
# we have natural C4 grasses for example. This setup is only good for deltas
defpfts = [14, 15, 16]

# Scenarios to plot deltas from reference time
usescens = ['rcp8.5-weg', 'rcp2.6-weg', 'rcp8.5-seg', 'rcp2.6-seg', 'rcp8.5', 'rcp2.6']

# Scenarios to calculate difference between them
diffscenhi = 'rcp8.5-weg'
diffscenlo = 'rcp8.5-seg'

# domain = "SAMATL" # South america and Atlantic Ocean
domain = "BR" # Zoom in Brazil

# Read regions? We have to if we want to make all deforestation plots
regfname =      "../regions/states_grid.nc"
regcodesfname = "../regions/states_codes.csv"
useregs = ["AM","PA","MT","MA","TO","PI"]


plotsfolder = "../output_plots/deforestation/" 

# plotfname =     plotsfolder + "deforestation_noref_" + domain
# diffplotfname = plotsfolder + "deforestation_noref_diff_" + domain
# wks_type = "png"

#%%
# Derived configurations

# Remove the 0 in the middle of deltalevels, so the white spans both positive and negative sides
# Do the same for eflevels, which are also anomalies
# deltalevels = np.delete(deltalevels,np.where(deltalevels == 0))
# eflevels = np.delete(eflevels,np.where(eflevels == 0))

# ==================================================== Domains
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



# %%
# ===================== Read data
allfnames = glob.glob(infolder + "*.nc")

# alldatadic = {i.attrs["defscenario"]:i for i in [xr.open_dataset(j)[pftvarname] for j in allfnames]}

# alldatadic = {i.attrs["defscenario"]:i for i in [xr.open_dataset(j)[pftvarname] for j in allfnames]}

# Read all input files in in a dataset using the "defscenario" 
# attribute added by fix_deforestation.py
def setvname(arr,name):
    arr.name = name
    return(arr)
    
bigds = xr.merge([setvname(i, i.attrs["defscenario"]) for i in \
    [xr.open_dataset(j)[pftvarname] for j in allfnames]])

#%% Read regions
regions = xr.open_dataarray(regfname).interp_like(bigds, method = "nearest").rename("region")

regcodes = pd.read_csv(regcodesfname).set_index("code").T.to_dict("list")
regcodes = {i:regcodes[i][0] for i in regcodes.keys()}

# Calculate deforestation
defds = bigds.isel(pft = defpfts).sum(dim = "pft")

# defdsdim = 

# Get a single datarray with a scenario dimension
defda = xr.concat([setvname(defds[vname],"deforestation").assign_coords({"scenario":vname}).expand_dims("scenario") for vname in defds.data_vars], dim="scenario")

allds = xr.merge([defda, regions])

alldf = allds.to_dataframe().replace({"region":regcodes}).groupby(["scenario","region","time"]).mean().reset_index()

alldf[alldf["region"].isin(useregs) and alldf["scenario"].isin(["veg"])]


# defda.mean(["lat","lon"]).to_dataframe()

#%% TESTING
(
    p9.ggplot(alldf[alldf["region"].isin(useregs)]) +
    p9.aes(x = "time", y = "deforestation", color = "scenario") + 
    p9.geom_line() +
    p9.facet_wrap("region")
)

#%%


# Reference array
refarr = setvname(defds[refscen].sel(time = refyear), "reference")

# Changes in def between pyear and refyear
deltads = defds.sel(time = pyear) - refarr

# Create output folder
os.makedirs(plotsfolder, exist_ok = True)

# %%
# ===================== BEGIN PLOTS

wksres = Ngl.Resources()
wksres.wkHeight = 2048
wksres.wkWidth = 2048
wks = Ngl.open_wks(wks_type,plotfname,wksres)  # Open a workstation.

# Function to set common resources
def set_common_resources():
    res = Ngl.Resources()

    res.nglDraw         =   False
    res.nglFrame         =   False

    res.sfXArray               = defds["lon"].values
    res.sfYArray               = defds["lat"].values

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

# Resources for reference contour/fill plots
refres = set_common_resources()

refres.cnFillOn               = True
refres.cnLinesOn              = False
refres.cnLineLabelsOn         = False
refres.cnFillMode             = "RasterFill"
# refres.lbLabelBarOn           = False

refcolormap = Ngl.read_colormap_file("matlab_jet")
# refcolormap = refcolormap[::-1]
refres.cnFillPalette           =   refcolormap


# refres.cnLevelSelectionMode    =   "ExplicitLevels"
# refres.cnLevels    =   reflevels

# Reference plot
refplot = Ngl.contour_map(wks,refarr.to_masked_array(),refres)

# =============================== Delta plots
# Resources for delta plots
deltares = set_common_resources()

deltares.cnFillOn               = True
deltares.cnLinesOn              = False
deltares.cnLineLabelsOn         = False
deltares.cnFillMode             = "RasterFill"
deltares.lbLabelBarOn           = False

# deltacolormap = Ngl.read_colormap_file("WhiteYellowOrangeRed")
deltacolormap = Ngl.read_colormap_file("OceanLakeLandSnow")
deltacolormap = deltacolormap[2:-24]
# deltacolormap = deltacolormap[::-1]
deltares.cnFillPalette           =   deltacolormap

deltares.cnLevelSelectionMode    =   "ExplicitLevels"
deltares.cnLevels    =   np.append(np.array([5]),np.arange(10,100.1,10))

# Plots
plots = []
figstrs = []
for usescen in usescens:
    # pltarray = deltads[usescen]
    pltarray = defds[usescen].sel(time = pyear)
    
    contplot = Ngl.contour_map(wks,pltarray.to_masked_array(),deltares)
    plots.append(contplot)
    figstrs.append(usescen)
    # figstrs.append("historical | " + monstrs[usemon])

panelres                                  = Ngl.Resources()
panelres.nglPanelFigureStrings            = figstrs
panelres.nglPanelFigureStringsFontHeightF = 0.01
panelres.nglPanelLabelBar                 = True     # Turn on panel labelbar
panelres.nglPanelLabelBarLabelFontHeightF = 0.01    # Labelbar font height
panelres.nglPanelLabelBarHeightF          = 0.03   # Height of labelbar
# panelres.nglPanelLabelBarWidthF           = 0.700    # Width of labelbar
# panelres.nglPanelTop                      = 0.5
panelres.nglPanelBottom                      = 0.05
# panelres.nglPanelRight                      = 0.2

# panelres.lbTitleString          =   reflabelstring
panelres.lbTitleString          =   "Total deforestation in  " + str(pyear) + " (pp)"
panelres.lbTitlePosition          =   "Bottom"
panelres.lbTitleFontHeightF          =   0.01
panelres.lbJustification          = "TopCenter"

# panelres.nglFrame = False

Ngl.panel(wks,plots,[3,2],panelres)


# Ngl.draw(refplot)
# Ngl.frame(wks)

Ngl.delete_wks(wks)

# %%
# ================================== Scenario difference plots

wksres = Ngl.Resources()
wksres.wkHeight = 2048
wksres.wkWidth = 2048
wks = Ngl.open_wks(wks_type,diffplotfname,wksres)  # Open a workstation.

# Resources for diff plots
diffres = set_common_resources()

diffres.cnFillOn               = True
diffres.cnLinesOn              = False
diffres.cnLineLabelsOn         = False
diffres.cnFillMode             = "RasterFill"
diffres.lbLabelBarOn           = False
# diffres.lbOrientation          = "horizontal"
# diffres.lbTitleString          = "Difference in " + str(pyear) + " deforestation (" + diffscenhi + " - " + diffscenlo + ")"

diffcolormap = Ngl.read_colormap_file("WhiteYellowOrangeRed")
# diffcolormap = diffcolormap[::-1]
diffres.cnFillPalette           =   diffcolormap

diffres.cnLevelSelectionMode    =   "ExplicitLevels"
diffres.cnLevels    =   np.append(np.array([5]),np.arange(10,70.1,10))

diffarr = defds[diffscenhi].sel(time = pyear) - defds[diffscenlo].sel(time = pyear)

# Plots
baseplots = []
basefigstrs = []
for usescen in [diffscenhi, diffscenlo]:
    pltarray = deltads[usescen]
    contplot = Ngl.contour_map(wks,pltarray.to_masked_array(),deltares)
    baseplots.append(contplot)
    basefigstrs.append(usescen)

diffplot = Ngl.contour_map(wks,diffarr.to_masked_array(),diffres)

basepanelres = copy.deepcopy(panelres)
basepanelres.nglPanelFigureStrings = basefigstrs
basepanelres.nglFrame = False
basepanelres.nglPanelRight = 0.65
basepanelres.nglPanelLeft = 0.05

Ngl.panel(wks,baseplots,[1,2],basepanelres)

diffpanelres = copy.deepcopy(panelres)
diffpanelres.nglPanelFigureStrings = ["Difference"]
diffpanelres.nglPanelRight = 0.95
diffpanelres.nglPanelLeft = 0.65
diffpanelres.lbTitleString          = "Difference in " + str(pyear) + " deforestation~C~ (" + diffscenhi + " - " + diffscenlo + ")"

Ngl.panel(wks,[diffplot],[1,1],diffpanelres)


# Ngl.draw(diffplot)
# Ngl.frame(wks)

Ngl.delete_wks(wks)
# %%
