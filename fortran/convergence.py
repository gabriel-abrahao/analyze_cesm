#%%
import numpy as np
import xarray as xr
import metpy.calc
from metpy.units import units
# import pandas as pd
# import scipy.stats
# import Ngl
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

# from mpl_toolkits.basemap import Basemap
# import cartopy.crs as ccrs
# from cartopy.io.shapereader import Reader
# from cartopy.feature import ShapelyFeature
# import cftime
import os
import re

import f90wrap
# Uses f2py/f90wrap, must build first with make
import dpres
# import dpres.dpres

# import numba
# from numba import jit, vectorize, guvectorize
# import dask.array as da

#%%
# For testing
bigdsin = xr.open_dataset("../refdata/historical/historical_0.0_heg_ensmean_pres_1990_2005.nc")

#%%
def calc_moisture_convergence(bigdsin):
    # Subset only needed vars because metpy.parse_cf() iterates over all variables
    usevars = ["U", "V", "Q"]

    # We need to parse the CRS to metpy so it can calculate divergence in meters
    cfbigdsin = bigdsin[usevars].metpy.parse_cf()
    # cfbigdsin = bigdsin.metpy.parse_cf(usevars,{"latitude":"lat", "longitude":"lon"})

    # Add metpy units so the divergence function works properly
    u = cfbigdsin["U"] * units.meters / units.seconds
    v = cfbigdsin["V"] * units.meters / units.seconds
    q = cfbigdsin["Q"] * 1000.0 * units.grams / units.kilograms # kg/kg to g/kg

    # Moisture flux. Ignores turbulent horizontal flux
    uq = u*q
    vq = v*q

    # Moisture flux convergence (inverse of divergence)
    convq = -metpy.calc.divergence(uq,vq)

    # Reconvert to a DataArray without the metpy behavior
    convq = xr.DataArray(convq)
    convq.attrs["long_name"] = "Moisture convergence"
    convq.attrs["units"] = "g kg-1 s-1"
    convq.attrs["comment"] = "Calculated with V*Q and U*Q, thus ignoring turbulent fluxes"
    return(convq)

calc_moisture_convergence(bigdsin)


# %%
# (dlon, dlat) = metpy.calc.lat_lon_grid_deltas(cfbigdsin["lon"], cfbigdsin["lon"])
# div = metpy.calc.divergence(u,v, dx = dlat, dy = dlon)

#%%
# xdlat =  xr.DataArray(dlat)
# xdlon =  xr.DataArray(dlon)




#%% 
plevel = np.array([ 1000.,950.,900.,850.,800.,750.,700.,650.,600., \
            550.,500.,450.,400.,350.,300.,250.,200., \
            175.,150.,125.,100., 80., 70., 60., 50., \
            40., 30., 25., 20. ])*100

# plevel = bigdsin.plev
klvl = len(plevel)
# psfc = 101525
psfc = 101800
# pmsg = min(plevel)
# pmsg = 3000 #???
pmsg = np.nan
ptop = min(plevel)
dp = np.zeros_like(plevel)
iopt = 0
kflag = 0
ier = 0

dpres.dpres.dpres1d(klvl,plevel,psfc,pmsg,ptop,dp,iopt,kflag,ier)
dp