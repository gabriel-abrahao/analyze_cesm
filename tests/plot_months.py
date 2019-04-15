import numpy as np
import xarray as xr
import Ngl
import matplotlib.pyplot as plt

inpfname = "/home/gabriel/apagar/rcp8.5_weg_005.cam2.h1.2050-01-01-00000.nc"

pdffname = "test.pdf"

fin = xr.open_dataset(inpfname)

var = fin['TREFHT']

maxlat = 5.0
minlat = -33.0
maxlon = 326.0
minlon = 280.0

var = var.sel(lat = slice(minlat,maxlat), lon = slice(minlon,maxlon))

var = var.isel(time = 0)

print(var)

# var.plot()
# plt.figure()
xr.plot.pcolormesh(var, levels = range(290,310,2), cmap = "jet")
plt.title("lala")
plt.savefig(pdffname)
# plt.show()
