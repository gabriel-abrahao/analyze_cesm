import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats
# import Ngl
import matplotlib.pyplot as plt
import matplotlib
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

    pdffname = "tests/periods_" + inpvname + ".pdf"

    maxlat = 5.0
    minlat = -33.0
    maxlon = 326.0
    minlon = 280.0

    # Periods for the t-test
    p1years = (2013,2031)
    p2years = (2032,2050)

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

    # Convert precipitation units
    if allinp.attrs['units'] == 'm/s':
        allinp.values = allinp.values*86400000.0
        allinp.attrs['units'] = 'mm/day'
        cmap = "RdBu"
    else:
        cmap = "RdBu_r"



    # FIXME: For some reason, CDO can put an extra value for the first month
    # This fix removes it if the input has an odd number of values
    if np.mod(allinp.coords['time'].shape,2) != 0.0:
        print("WARNING: Odd number of times in variable, dropping the first...")
        allinp = allinp.isel(time = slice(1,None))

    id = pd.MultiIndex.from_arrays([allinp.coords['time.month'].values, allinp.coords['time.year'].values], names=['month','year'])
    allinp.coords['time'] = id
    allinp = allinp.unstack('time')

    allp1 = allinp.sel(year = slice(*p1years))
    allp2 = allinp.sel(year = slice(*p2years))

    # vmax = max(np.max(test_rcp85['diff']),np.max(test_rcp26['diff'])).values
    # vmin = min(np.min(test_rcp85['diff']),np.min(test_rcp26['diff'])).values
    # vmax = max(abs(vmax),abs(vmin))

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
    # fc = xr.plot.imshow(test_year_ens['diff'].isel(month = [0,1], scenario = [0,1]),row = 'month',col = 'scenario')
    # fc = xr.plot.imshow(test_year_ens['diff'],row = 'month',col = 'scenario', cbar_kwargs = {'label':inpvname})
    # # ax = fc.axes.flat[0]
    # # dir(fc)
    # # xr.plot.contourf(test_year_ens['diff'].isel(month = 0, scenario = 0),ax=ax,levels = [0,0.05,2],hatches=['','.'],alpha = 0)
    # for ax in fc.axes.flat:
    #     # ax.set_title('teste')
    #     temptitle = ax.get_title()
    #     xr.plot.contourf(test_year_ens['diff'].isel(month = 0, scenario = 0),ax=ax,levels = [0,0.05,2],hatches=['///','...',''],alpha = 0,add_colorbar=False)
    #     ax.set_title(temptitle)
    #     ax.set_xlabel('')
    #     ax.set_ylabel('')

    with PdfPages(pdffname) as pdf:
        # labelpref = allinp.attrs['long_name'] + '(' + allinp.attrs['units'] + ')' + '\n' + 'WEG-SEG ' + '(' + '-'.join([str(i) for i in pyears]) + ') '
        labelpref = allinp.attrs['long_name'] + '(' + allinp.attrs['units'] + ')' + '\n' + '(' + '-'.join([str(i) for i in p1years]) + ',' + '-'.join([str(i) for i in p1years]) + ') '
        fc_year_ens = plot_sig_facet(test_year_ens, row = 'month', col = 'scenario',cbar_kwargs = {'label':labelpref},cmap = cmap)
        pdf.savefig()
        plt.close()


    # fc = xr.plot.contourf(test_year_ens['diff'],row = 'month',col = 'scenario',levels = [0,0.05,2],hatches=['','.'],alpha = 0)

    # xr.plot.imshow(test_year_ens['diff'].isel(month = 0, scenario = 0))
    # xr.plot.contourf(test_year_ens['pvalue'].isel(month = 0, scenario = 0),levels = [0,0.05,0.10,2],hatches=['///','...',''],alpha = 0)
    # fc = xr.plot.imshow(test_year_ens['pvalue'].isel(month = 0, scenario = 0))
    # fc = xr.plot.contourf(test_year_ens['diff'].isel(month = 0, scenario = 0),levels = [0,0.05,2],hatches=['','.'],alpha = 0)

    # plt.savefig(fig,"foo.pdf", format = 'pdf', bbox_inches='tight')
    # plt.savefig(pdffname, format = 'pdf', bbox_inches='tight')
    # plt.show()

    # xr.plot.imshow(anom.isel(month = 0,scenario = 0))

    ds = xr.tutorial.open_dataset('air_temperature').load()
    t = ds.air.groupby('time.season').mean(dim='time')

    # allinp.sel(year = slice(*(2013,2030)))

    # test = allinp.isel(ensemble = 0, scenario = 1)
    # poi = test.coords['time'].values
    # poi
    # id = pd.MultiIndex.from_arrays([test.coords['time.month'].values, test.coords['time.year'].values], names=['month','year'])
    # test.coords['time'] = id
    # poi = test.coords['time'].values
    # poi.shape
    # np.unique(poi).shape
    # test.sel(month = 1)
    # test.unstack('time')

    # Math won't work if times are different
    # allp1.coords['time'] = allp2.coords['time']
    # anom = allp2-allp1
    # anomsig = scipy.stats.ttest_ind(allp2,allp1,axis=2) #Not quite right
    #
    #
    # poi = xr.DataArray(allp2.values - allp1.values, coords = allp2.coords)
    # poi

    # plot = xr.plot.imshow(anom.isel(time = 0), col = 'scenario', row = 'ensemble', cmap = "jet")
    # plot = xr.plot.imshow(anom.mean(dim = 'ensemble'), col = 'scenario', row = 'time', cmap = "jet")


    # v1 = allp1.isel(time = 0, scenario = 0, ensemble = 0)
    # sig = xr.where(v1 <= 290.0, 1.0, 0.0)
    #
    # xr.plot.imshow(v1, cmap = "jet")
    # xr.plot.contourf(sig, levels = [0,0.99,2],hatches=['','.'],alpha = 0)

    # plt.show()

    # hvplot.quadmesh(anom)


    # plot = xr.plot.imshow(allp1.isel(time = 0), col = 'scenario', row = 'ensemble', levels = np.arange(290.0,300.0,0.5), cmap = "jet")

    # plot = xr.plot.imshow(varall[:,0,:,:], col = 'run', col_wrap=4, levels = range(290,310,2), cmap = "jet")

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
