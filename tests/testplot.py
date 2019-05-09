import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats
import Ngl
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cftime
import os
import re

def main():
    maindir = 'input/allmon/'
    inpfsuf = "allmon.nc"
    # p1suf = 'mon_avg_2013_2031.nc'
    # p2suf = 'mon_avg_2032_2050.nc'
    # inpfname = "input/ymonmeans/rcp2.6_seg_005/all_mon.nc"

    inpvname = "TREFHT"

    shpfname = "/home/gabriel/shapefiles/estados_2010.shp"

    pdffname = "test.pdf"

    maxlat = 5.0
    minlat = -33.0
    maxlon = 326.0
    minlon = 280.0

    # Periods for the t-test
    p1years = (2013,2031)
    p2years = (2032,2050)

    allinp = xr.open_dataset("input/temp_allinp.nc")
    allinp = allinp[[i for i in allinp.data_vars.variables][0]]

    allp1 = allinp.sel(year = slice(*p1years))
    allp2 = allinp.sel(year = slice(*p2years))

    # We have to mess up the metadata
    # allp2.coords['year'] = allp1.coords['year']
    anom = allp2.mean('year') - allp1.mean('year')

    anom = allp2 - allp1

    # test_year = scipy.stats.ttest_ind(allp2,allp1,axis = allp1.dims.index('year'))
    test_year_ens = ttest(allp2,allp1,dims = ['year','ensemble'])
    test_year_ens


    # xr.plot.imshow(test_year_ens['statistic'].isel(month = 0,scenario = 0))
    # xr.plot.imshow(test_year_ens['diff'].isel(month = 0,scenario = 0))

    # fig = plt.figure()
    # fc = xr.plot.imshow(test_year_ens['diff'],row = 'month',col = 'scenario')
    # fc = xr.plot.imshow(test_year_ens['diff'].isel(month = [0,1], scenario = [0,1]),row = 'month',col = 'scenario')
    #
    #
    # # xr.plot.contourf(test_year_ens['diff'].isel(month = 0, scenario = 0),ax=ax,levels = [0,0.05,2],hatches=['','.'],alpha = 0)
    # for ax in fc.axes.flat:
    #     # ax.set_title('teste')
    #     xr.plot.contourf(test_year_ens['diff'].isel(month = 0, scenario = 0),ax=ax,levels = [0,0.05,2],hatches=['///','...',''],alpha = 0)
    #
    # ax = fc.axes.flat[0]
    # print(dir(ax))
    # print(plt.getp(ax))

    # fc = xr.plot.contourf(test_year_ens['diff'],row = 'month',col = 'scenario',levels = [0,0.05,2],hatches=['','.'],alpha = 0)

    # xr.plot.imshow(test_year_ens['diff'].isel(month = 0, scenario = 0))
    # xr.plot.contourf(test_year_ens['pvalue'].isel(month = 0, scenario = 0),levels = [0,0.05,0.10,2],hatches=['///','...',''],alpha = 0)
    # fc = xr.plot.imshow(test_year_ens['pvalue'].isel(month = 0, scenario = 0))
    # fc = xr.plot.contourf(test_year_ens['diff'].isel(month = 0, scenario = 0),levels = [0,0.05,2],hatches=['','.'],alpha = 0)


    # plt.savefig("foo.pdf", format = 'pdf', bbox_inches='tight')


    var = test_year_ens['diff'].isel(scenario = 0, month = 0).values
    sig = test_year_ens['pvalue'].isel(scenario = 0, month = 0).values
    lat = test_year_ens['lat'].values
    lon = test_year_ens['lon'].values

    valmin = np.min(var)
    valmax = np.max(var)
    if np.sign(valmax) != np.sign(valmin):
        valmaxabs = max([abs(valmin),abs(valmax)])
        valmax = valmaxabs
        valmin = -valmaxabs

    valmin = -0.7
    valmax = 0.7
    spacing = 0.1

    res                        = Ngl.Resources()
    res.nglFrame               = False
    res.nglDraw                = False

    res.sfXArray               = lon
    res.sfYArray               = lat

    res.mpLimitMode            = 'LatLon'
    res.mpMinLatF              = min(lat)
    res.mpMaxLatF              = max(lat)
    res.mpMinLonF              = min(lon)
    res.mpMaxLonF              = max(lon)

    res.mpGridAndLimbOn         = False

    # Resources specific to the color filling
    rescol                         =   res

    rescol.mpDataSetName           = "Earth..4"
    rescol.mpOutlineSpecifiers     = ["Brazil"]

    rescol.cnFillPalette          = "BlueDarkRed18"
    rescol.cnFillOn               = True
    rescol.cnFillMode             = "RasterFill"

    rescol.cnLevelSelectionMode   = "ExplicitLevels"
    rescol.cnLevels               = np.arange(valmin,valmax,spacing)


    rescol.cnLinesOn              = False
    rescol.cnLineLabelsOn         = False
    # rescol.cnInfoLabelOn          = False
    # rescol.lbOrientation          = "Horizontal"


    ressig                        = res
    ressig.cnFillOn               = True
    ressig.cnFillMode             = "AreaFill"
    # ressig.cnLinesOn              = True
    # ressig.cnLineLabelsOn         = True
    ressig.cnLevelSelectionMode   = "ExplicitLevels"
    ressig.cnLevels               = [0.05,0.10]

    ressig.cnMonoFillColor        = 1
    ressig.cnMonoFillPattern      = False
    ressig.cnFillPatterns         = [3,17,-1]


    wks_type = "pdf"
    wks = Ngl.open_wks(wks_type,"foo.pdf")

    plotcol = Ngl.contour_map(wks,var,rescol)
    plotsig = Ngl.contour_map(wks,sig,ressig)

    # Ngl.draw(plotcol)
    Ngl.draw(plotsig)
    Ngl.frame(wks)

    Ngl.end()


def ttest(inparr1,inparr2,dims):
    # inparr1 = allp2
    # inparr2 = allp1
    # # dims = ['year','ensemble']
    # dims = 'year'

    if not isinstance(dims,list): dims = [dims]

    # A name for the stacked dimension to be tested along
    testdimname = 'testdim'

    arr1 = inparr1.stack({testdimname : dims})
    arr2 = inparr2.stack({testdimname : dims})

    diff = arr1.mean(**{'dim' : testdimname}) - arr2.mean(**{'dim' : testdimname})

    # TODO: See what can be done about aligning
    ind = arr1.dims.index(testdimname)
    test = scipy.stats.ttest_ind(arr1,arr2,axis = ind)
    results = xr.Dataset()
    results['diff'] = diff
    results['statistic'] = xr.DataArray(test.statistic, coords = arr1.isel({testdimname : 0}).coords)
    results['pvalue'] = xr.DataArray(test.pvalue, coords = arr1.isel({testdimname : 0}).coords)
    results = results.drop(testdimname)
    results.attrs['ttest_dims'] = 't-test made along dimensions: ' + ','.join(dims)
    return results


if __name__ == '__main__':
    main()
