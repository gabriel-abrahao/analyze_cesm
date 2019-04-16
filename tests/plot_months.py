import numpy as np
import xarray as xr
import Ngl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cftime

inpfname = "/home/gabriel/apagar/rcp8.5_weg_005.cam2.h1.2050-01-01-00000.nc"

shpfname = "/home/gabriel/shapefiles/estados_2010.shp"

pdffname = "test.pdf"

fin = xr.open_dataset(inpfname)

var = fin['TREFHT']

maxlat = 5.0
minlat = -33.0
maxlon = 326.0
minlon = 280.0

# var = var.sel(lat = slice(minlat,maxlat), lon = slice(minlon,maxlon))

# var = var.isel(time = range(0,4))

print(var)
# print(var.coords['time.month'])
# print(var.sel(time = slice('2050-02-01','2050-02-05')))
# print(var.groupby('time.month').mean(dim = 'time'))

# var.plot()
# plt.figure()
# xr.plot.pcolormesh(var, levels = range(290,310,2), cmap = "jet", col='time', col_wrap = 2)
# plot = xr.plot.pcolormesh(var[0,:,:], levels = range(290,310,2), cmap = "jet")
# poi = plot.axes.xaxis.get_majorticklabels()
# print(var.coords['lon'])

# wrap_var =

# ax = plt.axes()
shape_feature = ShapelyFeature(Reader(shpfname).geometries(),
                                ccrs.PlateCarree())

# [print(i) for i in Reader(shpfname).geometries()]

# poi = list(Reader(shpfname).geometries())

ax = plt.axes(projection=ccrs.PlateCarree())

# ax.add_feature(shape_feature)
# plot = xr.plot.pcolormesh(var[0,:,:], ax = ax, levels = range(290,310,2), cmap = "jet", transform=ccrs.PlateCarree())
plot = xr.plot.imshow(var[0,:,:], ax = ax, levels = range(290,310,2), cmap = "jet", transform=ccrs.PlateCarree())
# plot = xr.plot.imshow(var[0,:,:], ax = ax, levels = range(290,310,2), cmap = "jet", transform=ccrs.PlateCarree(central_longitude=180.0))
# ax.set_extent((minlon,maxlon,minlat,maxlat))
# ax.set_xlim((minlon-360,maxlon-360))
# ax.set_xlim((minlon,maxlon))
# ax.set_ylim((minlat,maxlat))

ax.add_feature(shape_feature)

# ax.coastlines()
# print(ax.projection)

# map = Basemap(projection = "cyl")
# map.drawcountries()
# map.contourf(var[0,:,:], ax = ax, levels = range(290,310,2), cmap = "jet")


# plt.title("lala")
plt.savefig(pdffname)
# plt.show()
