# Burns agricultural tables per municipalities to rasters
# WARNING: Should be used only with intensive variables,
# and even then the rasterization method does not guarantee
# the maintenance of sums at any level
# Errors for Brazil in a 0.9x1.25 grid are about 3% (over) on
# national soy area sum and 5% (under) for total production for 2016
# Total areas/production are recovered via area fractions and yields
# using cell (pixel) areas
#%%
# import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd 
import fiona
import rasterio
import rioxarray
import affine

import tqdm
import os

import xarray as xr
import matplotlib.pyplot as plt

#%%
agfname = "../../external/agroserv-stat-model-db/joined/joined_dataset_v1.csv"
# inshpfname = "../../external/agroserv-stat-model-db/input_shape/remade_munic/municipios_2015_remade.shp"
inshpfname = "../../external/agroserv-stat-model-db/input_shape/municipios_2015_WGS84.shp"
# os.listdir(inshpfname)

# reffname = "malha_estado_sam_ref.nc"
reffname = "../states_grid.nc"
# reffname = "/media/gabriel/hd1_6tb/backup/gabriel/doutorado/artigo_atribuicao/observados_v4/original_aninha/LANDSOYBEAN_GRAIN_2012.nc"
# reffname = "/media/gabriel/hd1_6tb/backup/gabriel/doutorado/artigo_atribuicao/observados_v4/maryland_km2_1grau.nc"

outfname = "../agdata.nc"

# Harvest years
syear = 1974
# syear = 2014
eyear = 2016

# Tuples with each crop's long name and variable suffix (as in the columns of agfname)
croptuples = [("Soybean", "s"), ("2nd crop Maize", "m2")]


#%%
# Gets a tuple with (latname,lonname) from a DataArray or DataSet
# by testing several possibilities 
def get_lat_lon_name(da):
    latopts = ["latitude", "lat", "y"]
    lonopts = ["longitude", "lon", "x"]
    latname = [i for i in da.coords if i in latopts][0]
    lonname = [i for i in da.coords if i in lonopts][0]
    return((latname,lonname))

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
# rasterize_to_ref(shp, refxr, trans, varname ,year, attrs).plot()
#%%
# Read munic shapefile
inshp = gpd.read_file(inshpfname)
inshp = inshp.rename({"CD_GEOCMU" : "codmun"}, axis=1)
inshp["codmun"] = inshp["codmun"].astype("string")

# Read Dataframe with all years of ag data
allagdata = pd.read_csv(agfname)
allagdata["codmun"] = allagdata["codmun"].astype("string")

# Calculate fractions
allagdata["shareafrac"] = allagdata["sharea"]/allagdata["areamun"]
allagdata["m2hareafrac"] = allagdata["m2harea"]/allagdata["areamun"]

#%%
# Open reference NetCDF
# Also open it with the rasterio interface to get the Affine transformation
refxr = xr.open_dataarray(reffname)
refmeta = xr.open_rasterio(reffname).attrs

trans = affine.Affine(*refmeta["transform"])

# Calculate pixel areas for the grid
pixelarea = get_pixel_areas(refxr)
#%%
# Set up functions for rasterization and run them in comprehensions
def rasterize_all_year_crop(shp, refxr, trans, year, croptuple):
    cropstr, croppref = croptuple

    varname = croppref + "hareafrac"
    attrs = {"units" : "frac", "long_name" : cropstr + " harvested area fraction"}
    hareafracda = rasterize_to_ref(shp, refxr, trans, varname ,year, attrs)
    hareafracda = nan_to_zero(hareafracda)
    hareafracda.name = varname

    varname = croppref + "yield"
    attrs = {"units" : "t/ha", "long_name" : cropstr + " harvested area fraction"}
    yieldda = rasterize_to_ref(shp, refxr, trans, varname ,year, attrs)
    yieldda = nan_to_zero(yieldda)
    yieldda.name = varname

    varname = croppref + "harea"
    attrs = {"units" : pixelarea.attrs["units"], "long_name" : cropstr + " harvested area"}
    hareada = hareafracda*pixelarea
    hareada.attrs = attrs
    hareada.name = varname

    varname = croppref + "tprod"
    attrs = {"units" : "tonnes", "long_name" : cropstr + " total production"}
    tprodda = hareada*yieldda*100.0
    tprodda.attrs = attrs
    tprodda.name = varname

    yearda = xr.merge([hareafracda, yieldda, hareada, tprodda])
    return(yearda)

# Select a single year and merge its agdata with a shapefile
def merge_year_shape(inshp, allagdata, year):
    # Filter year
    yagdata = allagdata.query('year == ' + str(year))
    # Merge with geodata
    shp = inshp.merge(yagdata, on = "codmun")
    return(shp)

def rasterize_all_year(shp, refxr, trans, year, croptuples):
    outds = xr.merge([rasterize_all_year_crop(shp, refxr, trans, year, croptuple) for croptuple in croptuples])
    return(outds)

#%% Actual calculations

yearsrange = tqdm.trange(syear,eyear+1, desc = "Rasterizing", leave = True)
agdslist = []
for year in yearsrange:
    yearsrange.set_description("Rasterizing %i" % year, refresh=True)
    agdslist.append(rasterize_all_year(merge_year_shape(inshp, allagdata, year), refxr, trans, year, croptuples) )

#%%
# Concatenate on the year dimension
agds = xr.concat(agdslist, dim = "year")

# Add pixelarea 
agds = agds.merge(pixelarea)

# Write output
agds.to_netcdf(outfname)
