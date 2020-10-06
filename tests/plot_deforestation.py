#%%
import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats
import Ngl
# import matplotlib.pyplot as plt
import matplotlib
# from matplotlib.backends.backend_pdf import PdfPages
#
# # from mpl_toolkits.basemap import Basemap
# import cartopy.crs as ccrs
# from cartopy.io.shapereader import Reader
# from cartopy.feature import ShapelyFeature
import cftime
import os
import re
import copy
#%%

# maindir = '../input/out_surfdata/'
maindir = '../input/out_surfdata_fix/'
inpfpref = "surfdata.pftdyn_0.9x1.25_"
# p1suf = 'mon_avg_2013_2031.nc'
# p2suf = 'mon_avg_2032_2050.nc'
# inpfname = "input/ymonmeans/rcp2.6_seg_005/all_mon.nc"


dicfnames = {
    'rcp2.6':'surfdata.pftdyn_0.9x1.25_rcp2.6_simyr1850-2100_c100323.nc',
    'rcp8.5':'surfdata.pftdyn_0.9x1.25_rcp8.5_simyr1850-2100_c100319.nc',
    'seg':'surfdata.pftdyn_0.9x1.25_rcp2.6_simyr1850-2100_rochedo_seg_c100323.nc',
    'weg':'surfdata.pftdyn_0.9x1.25_rcp2.6_simyr1850-2100_rochedo_weg_c100323.nc',
    'veg':'surfdata.pftdyn_0.9x1.25_rcp2.6_simyr1850-2100_veg_c100323.nc'
    }

inpvname = "PCT_PFT"

shpfname = "/home/gabriel/shapefiles/estados_2010.shp"

pdffname = "tests/landuse.pdf"

years = [2005,2030,2050]

maxlat = 5.0
minlat = -33.0
maxlon = 326.0
minlon = 280.0

    # A mask containing biome codes
    mskfname = "input/bioma.nc"
    mskvname = "const"

    mskarr = xr.open_dataset(mskfname)[mskvname].sel(lat = slice(minlat,maxlat), lon = slice(minlon,maxlon)).isel(time=0)
    allarrays = {}
    for scenario in dicfnames:
        fname = maindir + dicfnames[scenario]
        arr = xr.open_dataset(fname)[inpvname]
        # arr['lsmlat'] = ((xr.open_dataset(fname)['LATN'] + xr.open_dataset(fname)['LATS'])/2.0).mean(dim = 'lsmlon')
        # arr['lsmlon'] = ((xr.open_dataset(fname)['LONE'] + xr.open_dataset(fname)['LONW'])/2.0).mean(dim = 'lsmlat')
        # arr = arr.rename(dict(zip(list(arr.dims),[i[3:6] if i[0:3] == 'lsm' else i for i in list(arr.dims)])))
        arr = arr.sel(time = years, lat = slice(minlat,maxlat), lon = slice(minlon,maxlon))
        arr.name = scenario
        # allarrays[scenario] = lonflip(arr)
        allarrays[scenario] = arr

    # FIXME: This is a workaround for when lat lon of mskarr do not line up
    # exactly with the data, may cause problems
    for d in mskarr.dims:
        mskarr[d].values = list(allarrays.values())[0][d].values

    # Here we define who will be the reference vegetation to calculate deforestation
    # For now, deforestation will be the inverse of fraction of all forest and shrub PFTs
    # In relation to potential vegetation
    refarr = allarrays['veg'].sel(time = 2005).isel(pft=slice(1,11)).sum(dim="pft")
    defarrays = {}
    for scenario in dicfnames:
        defarrays[scenario] = 1-allarrays[scenario].isel(pft=slice(1,11)).sum(dim="pft")/refarr
        defarrays[scenario] = xr.apply_ufunc(np.nan_to_num,defarrays[scenario]) # Divide by zero yield zero
        defarrays[scenario] = xr.where(defarrays[scenario] < 0,0,defarrays[scenario])
        defarrays[scenario] = xr.where(defarrays[scenario] > 1,1,defarrays[scenario])
    # Begin the NGL part
    wks_type = "pdf"

    wks = Ngl.open_wks(wks_type,pdffname)

    res = Ngl.Resources()


    res.nglDraw = False
    res.nglFrame = False

    res.cnFillOn             = True
    res.cnFillMode  = 'RasterFill'
    res.cnFillPalette         = 'WhBlGrYeRe'
    res.cnLinesOn            = False
    res.cnLineLabelsOn       = False

    res.pmLabelBarDisplayMode = "Never"

    res.cnLevelSelectionMode = "ManualLevels"
    res.cnMinLevelValF       = 0.1
    res.cnMaxLevelValF       = 1.0
    res.cnLevelSpacingF      = 0.1

    res.tmXBOn = False
    res.tmXTOn = False
    res.tmYLOn = False
    res.tmYROn = False
    res.tmXBBorderOn = False
    res.tmXTBorderOn = False
    res.tmYLBorderOn = False
    res.tmYRBorderOn = False
    res.mpGridAndLimbOn = False

    res.cnConstFLabelOn = False
    res.cnConstFEnableFill = True

    res.mpOutlineBoundarySets = "National"
    res.mpDataSetName         = "Earth..4"
    res.mpDataBaseVersion     = "MediumRes"
    res.mpOutlineOn           = True
    res.mpOutlineSpecifiers   = ["Brazil:states"]

    # res.mp

    res.mpLimitMode = 'LatLon' # Needed in NGL for some reason
    res.mpMaxLatF = maxlat
    res.mpMinLatF = minlat
    res.mpMaxLonF = maxlon
    res.mpMinLonF = minlon

    res.sfXArray = list(allarrays.values())[0].lon.values
    res.sfYArray = list(allarrays.values())[0].lat.values

    # year = 2005
    # scenario = 'weg'

    plots = []
    for scenario in dicfnames:
        for year in years:
            res.tiMainString = scenario + str(year)
            plot = Ngl.contour_map(wks,defarrays[scenario].sel(time=year).values,res)
            plots.append(plot)

    nscenarios = len(dicfnames)
    nyears = len(years)

    respanel = Ngl.Resources()
    respanel.nglPanelLabelBar = True

    Ngl.panel(wks,plots,[nscenarios,nyears],respanel)

    # Ngl.draw(plot.base)
    # Ngl.frame(wks)
    print(defarrays['weg'].lon)
    Ngl.end()

    # defarrays[scenario].groupby(mskarr).mean(dim=['stacked_lat_lon']).plot()


