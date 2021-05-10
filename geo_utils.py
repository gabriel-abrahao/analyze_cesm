#%%
# import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd 
import fiona
import rasterio
import rioxarray
import affine
import copy

import tqdm
import os

import xarray as xr
import matplotlib.pyplot as plt

#%%
# Gets a tuple with (latname,lonname) from a DataArray or DataSet
# by testing several possibilities 
def get_lat_lon_name(da):
    latopts = ["latitude", "lat", "y"]
    lonopts = ["longitude", "lon", "x"]
    latname = [i for i in da.coords if i in latopts][0]
    lonname = [i for i in da.coords if i in lonopts][0]
    return((latname,lonname))

def get_lon_name(d):
    return(get_lat_lon_name(d)[1])

# Flipped here means that it goes from -180 to 180
def global_is_lon_flipped(d):
    lonname = get_lon_name(d)
    return not not np.any(d[lonname]<0.0).data # The double negation converts it to a python boolean

# Just works from a nonflipped (0:360) global grid
def global_lon_flip(d):
    lonname = get_lon_name(d)
    nlon = d[lonname].size
    dflip = copy.deepcopy(d.roll(lon=int(nlon/2),roll_coords=False))
    # dflip[lonname] = dflip[lonname] - 180.0
    dflip = dflip.assign_coords({lonname : dflip[lonname] - 180.0})
    dflip = dlip.assign_
    return dflip

# 
def global_lon_unflip(d):
    lonname = get_lon_name(d)
    nlon = d[lonname].size
    dflip = copy.deepcopy(d.roll(lon=int(nlon/2),roll_coords=False))
    # dflip[lonname] = dflip[lonname] + 180.0
    dflip = dflip.assign_coords({lonname : dflip[lonname] + 180.0})
    return dflip

# %%
def brazil_lon_unflip(d):
    dflip = copy.deepcopy(d)
    lonname = get_lon_name(d) 
    # lons = d[lonname].data
    # dflip[lonname].data = lons-360
    dflip = dflip.assign_coords({lonname : dflip[lonname] - 360.0})
    return dflip

def brazil_lon_flip(d):
    dflip = copy.deepcopy(d)
    lonname = get_lon_name(d) 
    # lons = d[lonname].data
    # dflip[lonname].data = lons-360
    dflip = dflip.assign_coords({lonname : dflip[lonname] + 360.0})
    return dflip

# Replaces np.nan with zeros in a Dataarray
def nan_to_zero(da):
    da = xr.where(np.isnan(da), 0.0, da)
    return(da)

# Calculate pixel areas on a regular lat/lon grid
# returns a DataArray broadcast to the same shape as the input
def get_pixel_areas(da):
    # (L) is the length measure used here
    unit = "km"
    P= 6356.7523 #Polar radius (L)
    E= 6378.1370 #Equatorial radius (L)

    latname, lonname = get_lat_lon_name(da)

    latitude = da[latname].values
    longitude = da[lonname].values

    nlat = latitude.shape[0]
    nlon = longitude.shape[0]

    reslat = latitude[1] - latitude[0]
    reslon = longitude[1] - longitude[0]

    # Convert to radians
    lat_new= (np.pi * latitude)/180.0

    # Polar circunference (L)
    PC = 2 * np.pi * P

    # Arc on latitude (L)
    dy = PC/(360.0/reslat)

    # Radius and circunference at each latitude
    Rlats = E * np.cos(lat_new)
    Clats = 2 * np.pi * Rlats 

    # Arc longitude
    dx = Clats/(360/reslon)

    arealats = dx*dy
    arealats = xr.DataArray(arealats, dims=latname).assign_coords({latname : da[latname]})

    areada = arealats.broadcast_like(da)

    areada.attrs["units"] = unit + "2"
    areada.attrs["long_name"] = "Pixel area"
    areada.name = "pixelarea"

    return(areada)


def rasterize_to_ref(shp, refxr, trans, varname, year, attrs, defval = np.nan):
    outxr = refxr.copy()
    outxr.values = np.where(outxr.values == 0, defval, defval)
    latname, lonname = get_lat_lon_name(outxr)

    outxr[latname][1] - outxr[latname][0]

    shapes = ((geom,value) for geom, value in zip(shp["geometry"], shp[varname]))

    burned = rasterio.features.rasterize(shapes=shapes, fill=defval, default_value=defval, out=outxr.values, transform=trans)
    burned = np.where(burned ==-9999, defval, burned)

    outxr = outxr.expand_dims("year").assign_coords({"year" : [year]})
    outxr.attrs = attrs
    outxr.name = varname

    if np.sign(outxr[latname][1] - outxr[latname][0]) > 0:
        outxr.values = np.flip(outxr.values, axis=outxr.get_axis_num(latname))

    return(outxr)

def read_shapefile(shpfname,**kwargs):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' 
    column holding the geometry information. This uses the pyshp
    package
    """
    sf_shape = shapefile.Reader(bioshpfname, **kwargs)

    fields = [x[0] for x in sf_shape.fields][1:]
    records = [y[:] for y in sf_shape.records()]
    #records = sf_shape.records()
    shps = [s.points for s in sf_shape.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df

def add_shapefile_df_polygons(wks,plotobj,shp,res=None):
    # tcoords = next(iter(shp["coords"]))
    for tcoords in shp["coords"]:
        (tx, ty) = tuple(zip(*tcoords))
        Ngl.add_polygon(wks, plotobj,tx,ty,res)