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
reffname = "../refdata/historical/historical_0.0_heg_ensmean_pres_1990_2005.nc"

plotfname = "../output_plots/geopotential_wind_v1"
wks_type = "png"

minlat = -61
maxlat = 15
minlon = 273
maxlon = 345


#%%
# Assume that there is a folder with each scenario, and right inside 
# it is a file matching insuf
scens = os.listdir(baseinfolder)
allfnames = [glob.glob(baseinfolder + i + "/" + "*" + insuf) for i in scens]
allfnames = [j for i in allfnames for j in i]

allitems = [{"fname" : fname, "ds" : xr.open_dataset(fname)} for fname in allfnames]
# allitems = [{"fname" : fname, "ds" : xr.open_dataset(fname).sel(lat = slice(minlat,maxlat), lon = slice(minlon,maxlon))} for fname in allfnames]
for item in allitems:
    item["ds"] = item["ds"].expand_dims("scenario")
    item["ds"]["scenario"] = pd.Index([os.path.basename(os.path.split(item["fname"])[0])])

bigdsin = xr.combine_nested([i["ds"] for i in allitems], concat_dim="scenario", combine_attrs= "override")

# %%
# Read reference file, and make sure metada matches
refds = xr.open_dataset(reffname)#.sel(lat = slice(minlat,maxlat), lon = slice(minlon,maxlon))
refds = refds.rename_dims(plev = "lev").rename_vars(plev = "lev")

refds = refds.assign_coords(lat = bigdsin["lat"].data)
refds = refds.assign_coords(lon = bigdsin["lon"].data)

# %%
# Delta between scenarios and reference
deltads = bigdsin - refds

# %%
# ===================== BEGIN PLOTS
wks = Ngl.open_wks(wks_type,plotfname)  # Open a workstation.

# Here we coarsen the wind array so the map doesn't get cramped
refbasedswind = refds.coarsen(lat = 2, lon = 2, boundary = "trim").mean()

# Function to set common resources
def set_common_resources():
    res = Ngl.Resources()

    res.nglDraw         =   False
    res.nglFrame         =   False

    res.sfXArray               = refds["lon"].values
    res.sfYArray               = refds["lat"].values

    res.vfXArray               = refbasedswind["lon"].values
    res.vfYArray               = refbasedswind["lat"].values

    res.mpLimitMode     = "LatLon"  # Change the area of the map
    res.mpMinLatF       =  minlat   # viewed.
    res.mpMaxLatF       =  maxlat  
    res.mpMinLonF       =  minlon
    res.mpMaxLonF       =  maxlon

    return(res)

# Resources for contour plot
contres = set_common_resources()

contres.cnFillOn               = True
contres.cnLinesOn              = False
contres.cnLineLabelsOn         = False



reswind = set_common_resources()

reswind.vcFillArrowsOn          = True
reswind.vcRefMagnitudeF         = 30.0             # define vector ref mag
reswind.vcRefLengthF            = 0.02             # define length of vec ref
reswind.vcMinFracLengthF        = 0.3
reswind.vcMinDistanceF          = 0.02
reswind.vcRefAnnoOrthogonalPosF = -0.20
reswind.vcRefAnnoFontHeightF    = 0.005
# reswind.vcRefAnnoOn    = False


# reswind.vcRefMagnitudeF = 10
# reswind.vcRefLenF = 0.04
# reswind.vcFillArrowsOn          = True
reswind.vcGlyphStyle = "CurlyVector"
# reswind.vcGlyphStyle = "WindBarb"

usemon = 12
uselev = 85000

# Plots
contplot = Ngl.contour_map(wks,refds["Z3"].sel(lev = uselev, month = usemon).to_masked_array(),contres)
dswind = refbasedswind.sel(lev = uselev, month = usemon)
windplot = Ngl.vector(wks,dswind["U"].to_masked_array(),dswind["V"].to_masked_array(),reswind)

Ngl.overlay(contplot,windplot)
Ngl.maximize_plot(wks, contplot)
Ngl.draw(contplot)
Ngl.frame(wks)

Ngl.delete_wks(wks)

# %% TEST
wks = Ngl.open_wks(wks_type,plotfname)  # Open a workstation.

dirc = Ngl.pynglpath("data")
f    = Nio.open_file(os.path.join(dirc,"cdf","uv300.nc"))
u    = f.variables["U"][1,:,:]
v    = f.variables["V"][1,:,:]
lat  = f.variables["lat"][:]
lon  = f.variables["lon"][:]
spd  = np.sqrt(u**2 + v**2)


cnres                             = Ngl.Resources()
cnres.nglDraw                     = False
cnres.nglFrame                    = False

cmap = Ngl.read_colormap_file("WhiteBlueGreenYellowRed")

# cnres.cnLinesOn                   = False
# cnres.cnLineLabelsOn              = False
# cnres.cnFillOn                    = True
# cnres.cnFillPalette               = cmap[15:,:]
# cnres.lbOrientation               = "horizontal"

# cnres.mpFillOn                    = False
# cnres.mpGeophysicalLineColor      = "Grey18"
# cnres.mpGeophysicalLineThicknessF = 1.5

# cnres.sfXArray                    = lon
# cnres.sfYArray                    = lat

vcres                         = Ngl.Resources()
vcres.nglDraw                 = False
vcres.nglFrame                = False

# vcres.vfXArray                = lon
# vcres.vfYArray                = lat

# vcres.vcFillArrowsOn          = True
# vcres.vcRefMagnitudeF         = 30.0             # define vector ref mag
# vcres.vcRefLengthF            = 0.02             # define length of vec ref
# vcres.vcMinFracLengthF        = 0.3
# vcres.vcMinDistanceF          = 0.02
# vcres.vcRefAnnoOrthogonalPosF = -0.20
# vcres.vcRefAnnoFontHeightF    = 0.005

# cnres.tiMainString    = "Fully opaque filled vectors over filled contours"

cnres.mpLimitMode     = "LatLon"  # Change the area of the map
cnres.mpMinLatF       =  minlat   # viewed.
cnres.mpMaxLatF       =  maxlat  
cnres.mpMinLonF       =  minlon
cnres.mpMaxLonF       =  maxlon

vcres.mpLimitMode     = "LatLon"  # Change the area of the map
vcres.mpMinLatF       =  minlat   # viewed.
vcres.mpMaxLatF       =  maxlat  
vcres.mpMinLonF       =  minlon
vcres.mpMaxLonF       =  maxlon

cnres.sfXArray               = refds["lon"].values
cnres.sfYArray               = refds["lat"].values

vcres.vfXArray               = refbasedswind["lon"].values
vcres.vfYArray               = refbasedswind["lat"].values

#---Draw fully opaque vectors
uv_plot  = Ngl.vector(wks,dswind["U"].to_masked_array(),dswind["V"].to_masked_array(),vcres)
spd_plot = Ngl.contour_map(wks,refds["Z3"].sel(lev = 85000, month = 12).to_masked_array(),cnres)

# uv_plot  = Ngl.vector(wks,u,v,vcres)
# spd_plot = Ngl.contour_map(wks,spd,cnres)
Ngl.overlay(spd_plot,uv_plot)
Ngl.maximize_plot(wks, spd_plot)
Ngl.draw(spd_plot)
Ngl.frame(wks)
Ngl.delete_wks(wks)



# %%
