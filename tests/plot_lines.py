import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cftime
import os
import re

inpfname = "output_small/allvars_biomes_monthly.nc"

inpvname = "TREFHT"

biome = 2.0 # Biome code, 2 = Cerrado, 5 = Amazon

finp = xr.open_dataset(inpfname)
varinp = finp[inpvname].sel(const = biome)

varinp.coords['time'] = pd.MultiIndex.from_arrays([varinp.coords['time.month'].values, varinp.coords['time.year'].values], names=['month','year'])
varinp.unstack('time')

mon = 1

colordict = {'rcp8.5_seg':'darkgreen','rcp8.5_weg':'darkred','rcp2.6_seg':'limegreen','rcp2.6_weg':'firebrick'}

means = varinp.mean(dim = 'ensemble')
maxes = varinp.max(dim = 'ensemble')
mines = varinp.min(dim = 'ensemble')

sce = 'rcp8.5_seg'

itime = varinp.sel(month = mon).coords['year']
imeans = means.sel(month = mon, scenario = sce).transpose()
imaxes = maxes.sel(month = mon, scenario = sce).transpose()
imines = mines.sel(month = mon, scenario = sce).transpose()
# plt.plot(itime,imaxes)
plt.fill_between(itime,imines,imaxes, color = colordict[sce], alpha = 0.2)
plt.plot(itime, imeans, color = colordict[sce])
# xr.plot.fill_between(maxes.sel(month = 1, scenario = 'rcp8.5_weg'), x = 'year')

# xr.plot.line(varinp.sel(month = 1, ensemble = '006'), x = 'year')


# xr.plot.line(varinp.sel(month = 1, ensemble = '006', scenario = 'rcp8.5_seg'), x = 'year', color = 'darkgreen')
# xr.plot.line(varinp.sel(month = 1, ensemble = '006', scenario = 'rcp8.5_weg'), x = 'year', color = 'darkred')
# xr.plot.line(varinp.sel(month = 1, ensemble = '006', scenario = 'rcp2.6_seg'), x = 'year', color = 'limegreen')
# xr.plot.line(varinp.sel(month = 1, ensemble = '006', scenario = 'rcp2.6_weg'), x = 'year', color = 'firebrick')

# xr.plot.line(varinp.sel(month = 1, scenario = 'rcp8.5_seg'), x = 'year', color = 'darkgreen')
# xr.plot.line(varinp.sel(month = 1, scenario = 'rcp8.5_weg'), x = 'year', color = 'darkred')
# xr.plot.line(varinp.sel(month = 1, scenario = 'rcp2.6_seg'), x = 'year', color = 'limegreen')
# xr.plot.line(varinp.sel(month = 1, scenario = 'rcp2.6_weg'), x = 'year', color = 'firebrick')


plt.show()
