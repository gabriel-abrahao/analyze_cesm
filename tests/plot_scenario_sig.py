import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats
# import Ngl
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

    inpvname = "PRECT"

    shpfname = "/home/gabriel/shapefiles/estados_2010.shp"

    pdffname = "tests/diffscenarios_" + inpvname + ".pdf"

    maxlat = 5.0
    minlat = -33.0
    maxlon = 326.0
    minlon = 280.0

    # Period to analyze
    pyears = (2040,2050)
    # p2years = (2032,2050)

    runnames = os.listdir(maindir)
    # runnames = ["rcp8.5_seg_006"]
    # inpfnames = {maindir + '/' + i + '/' + inpfsuf : i for i in runnames}
    # alldata = [{'p1fname': maindir + '/' + i + '/' + p1suf, 'p2fname': maindir + '/' + i + '/' + p2suf ,'runname' : i} for i in runnames]
    alldata = [{'inpfname': maindir + '/' + i + '/' + inpfsuf, 'runname' : i} for i in runnames]
    for item in alldata:
        item['inp'] = xr.open_dataset(item['inpfname'])[inpvname].sel(lat = slice(minlat,maxlat), lon = slice(minlon,maxlon))



    # Make sure the time axis are equal to the one in ref
    # ref = alldata[0]['p1'].coords['time']
    # for inpfname in inpfnames.keys():
    #     dicvars[inpfname].coords['time'] = ref
    #     dicvars[inpfname] = dicvars[inpfname].expand_dims()
    for item in alldata:
        sce, ens = item['runname'].rsplit('_',1)
        item['inp'] = item['inp'].expand_dims('scenario')
        item['inp'].coords['scenario'] = pd.Index([sce])
        item['inp'] = item['inp'].expand_dims('ensemble')
        item['inp'].coords['ensemble'] = pd.Index([ens])


    allinp = xr.merge([item['inp'] for item in alldata]).to_array().squeeze('variable')
    allinp.attrs = alldata[0]['inp'].attrs

    # FIXME: For some reason, CDO can put an extra value for the first month
    # This fix removes it if the input has an odd number of values
    if np.mod(allinp.coords['time'].shape,2) != 0.0:
        print("WARNING: Odd number of times in variable, dropping the first...")
        allinp = allinp.isel(time = slice(1,None))

    id = pd.MultiIndex.from_arrays([allinp.coords['time.month'].values, allinp.coords['time.year'].values], names=['month','year'])
    allinp.coords['time'] = id
    allinp = allinp.unstack('time')

    allinp = allinp.sel(year = slice(*pyears))

    # Convert precipitation units
    if allinp.attrs['units'] == 'm/s':
        allinp.values = allinp.values*86400000.0
        allinp.attrs['units'] = 'mm/day'
        cmap = "RdBu"
    else:
        cmap = "RdBu_r"

    test_rcp85 = ttest(allinp.sel(scenario = 'rcp8.5_weg'),allinp.sel(scenario = 'rcp8.5_seg'),dims = ['year','ensemble'])
    test_rcp26 = ttest(allinp.sel(scenario = 'rcp2.6_weg'),allinp.sel(scenario = 'rcp2.6_seg'),dims = ['year','ensemble'])

    # test_year = scipy.stats.ttest_ind(allp2,allp1,axis = allp1.dims.index('year'))
    # test_year_ens = ttest(allp2,allp1,dims = ['year','ensemble'])
    # test_year_ens
    # xr.plot.imshow(test_year_ens['statistic'].isel(month = 0,scenario = 0))
    # xr.plot.imshow(test_year_ens['diff'].isel(month = 0,scenario = 0))
    vmax = max(np.max(test_rcp85['diff']),np.max(test_rcp26['diff'])).values
    vmin = min(np.min(test_rcp85['diff']),np.min(test_rcp26['diff'])).values
    vmax = max(abs(vmax),abs(vmin))

    xr.plot.contourf(test_rcp85['pvalue'].isel(month = 0),levels = [0,0.01,0.05,2],hatches=['///','...',' '],alpha = 0)

    test_rcp85['pvalue'].isel(month = 0)

    with PdfPages(pdffname) as pdf:
        labelpref = allinp.attrs['long_name'] + '(' + allinp.attrs['units'] + ')' + '\n' + 'WEG-SEG ' + '(' + '-'.join([str(i) for i in pyears]) + ') '
        fc_rcp85 = plot_sig_facet(test_rcp85,col_wrap = 3, col = 'month', cbar_kwargs = {'label':labelpref + 'rcp85'},cmap = cmap, vmax = vmax)
        pdf.savefig()
        fc_rcp26 = plot_sig_facet(test_rcp26,col_wrap = 3, col = 'month', cbar_kwargs = {'label':labelpref + 'rcp26'},cmap = cmap, vmax = vmax)
        pdf.savefig()
        plt.close()

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

    try:
        ind = arr1.dims.index(testdimname)
    except:
        ind = [i[0] for i in arr1.dims.items()].index(testdimname)

    test = scipy.stats.ttest_ind(arr1,arr2,axis = ind)
    results = xr.Dataset()
    results['diff'] = diff
    results['statistic'] = xr.DataArray(test.statistic, coords = arr1.isel({testdimname : 0}).coords)
    results['pvalue'] = xr.DataArray(test.pvalue, coords = arr1.isel({testdimname : 0}).coords)
    results = results.drop(testdimname)
    results.attrs = arr1.attrs
    results.attrs['ttest_dims'] = 't-test made along dimensions: ' + ','.join(dims)
    return results

def plot_sig_facet(testresults,*args,**kwargs):
    facet = xr.plot.imshow(testresults['diff'],*args,**kwargs)

    matplotlib.rcParams['hatch.linewidth'] = 0.1

    for (i,j), ax in np.ndenumerate(facet.axes):
        temptitle = ax.get_title()
        namedict = (facet.name_dicts[i,j])
        # namedict will come up none if thats an empty tile
        if namedict is not None:
            xr.plot.contourf(testresults['pvalue'].sel(**namedict),ax=ax,levels = [0,0.01,0.05,2],hatches=['///','...',' '],alpha = 0,add_colorbar=False)
            # xr.plot.contourf(test_rcp85['pvalue'].sel(**namedict),ax=ax,levels = [0,0.05,2],hatches=['///','....',''],fill=False,add_colorbar=False)
            ax.set_title(temptitle)
            ax.set_xlabel('')
            ax.set_ylabel('')
    return facet


if __name__ == '__main__':
    main()
