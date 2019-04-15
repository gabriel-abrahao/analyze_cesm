import numpy as np
import xarray as xr
import Ngl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs

inpfname = "/home/gabriel/apagar/rcp8.5_weg_005.cam2.h1.2050-01-01-00000.nc"

pdffname = "test.pdf"

fin = xr.open_dataset(inpfname)

var = fin['TREFHT']

maxlat = 5.0
minlat = -33.0
maxlon = 326.0
minlon = 280.0

var = var.sel(lat = slice(minlat,maxlat), lon = slice(minlon,maxlon))

var = var.isel(time = range(0,4))

print(var)

# var.plot()
# plt.figure()
# xr.plot.pcolormesh(var, levels = range(290,310,2), cmap = "jet", col='time', col_wrap = 2)
# plot = xr.plot.pcolormesh(var[0,:,:], levels = range(290,310,2), cmap = "jet")
# poi = plot.axes.xaxis.get_majorticklabels()
print(var.coords['lon'])

# wrap_var =

# ax = plt.axes()
ax = plt.axes(projection=ccrs.PlateCarree())
xr.plot.pcolormesh(var[0,:,:], ax = ax, levels = range(290,310,2), cmap = "jet", transform=ccrs.PlateCarree())
ax.set_xlim((minlon-360,maxlon-360))
ax.set_ylim((minlat,maxlat))
# ax.coastlines()
# print(ax.projection)

# map = Basemap(projection = "cyl")
# map.drawcountries()
# map.contourf(var[0,:,:], ax = ax, levels = range(290,310,2), cmap = "jet")


# plt.title("lala")
plt.savefig(pdffname)
# plt.show()
