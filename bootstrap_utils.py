# %%
from matplotlib import pyplot as plt
import importlib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import ttest_ind_from_stats

import plotnine as p9

# from sklearn.linear_model import LinearRegression
# import statsmodels.api as sm
import functools

import cftime
import os
import glob
import re
import copy
import datetime
import numba
from numba import jit,prange
import sys

sys.path.append("../")
import pooled_stats 

#%%
# Calculates the slope of a regression line on the input n x 2 array
@jit(nopython=True)
def calc_slope(dfmat):
    x = dfmat[:,0]
    y = dfmat[:,1]
    if np.sum(np.isnan(y)) > 3:
        slope = np.nan
        # slope = np.sum(np.isnan(y))
    else:        
        # X = np.vstack([x, np.ones(len(x))]).T
        X = np.vstack((x, np.ones_like(y))).T
        # Perform the regression, betas come in inverse order (i.e. the last element is the constant)
        betas = np.linalg.inv(X.T @ X) @ X.T @ y.T
        slope = betas[0]
    return(slope)

def calc_slope_wrap(df,timeindex,regvarindex):
    dfmat = df.iloc[:,[timeindex,regvarindex]].to_numpy().astype("float64")
    # print(dfmat)
    return(calc_slope(dfmat))

# Calculates mean and variance of the slope of a regression line on the input n x 2 array
# using a nsamp bootsrap
@jit(nopython=True, fastmath=False)
def calc_slope_bootstrap(dfmat, nsamp = 30):
    xin = dfmat[:,0]
    yin = dfmat[:,1]

    ylen = len(yin)

    # Preallocate a vector to contain outputs
    # TODO: A running variance implementation would be more efficient
    # e.g. https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    slopes = np.empty(nsamp)

    for i in np.arange(nsamp):
        inds = np.random.choice(np.arange(ylen), ylen)
        
        x = np.take(xin,inds)
        y = np.take(yin,inds)

        X = np.vstack((x, np.ones_like(y))).T
        det = np.linalg.det(X.T @ X) # Determinant to catch singular matrices

        if (np.sum(np.isnan(y)) > 3) | (det == 0):
            slope = np.nan
            # slope = np.sum(np.isnan(y))
        else:        
            # Perform the regression, betas come in inverse order (i.e. the last element is the constant)
            betas = np.linalg.inv(X.T @ X) @ X.T @ y.T
            slope = betas[0]
        slopes[i] = slope
    # print(slopes)
    out = np.array((np.mean(slopes), np.var(slopes)))
    # print(out)
    return(out)

@jit(nopython = True)
def calc_slope_bootstrap_wrap(df,timeindex,regvarindex, nsamp = 30):
    dfmat = df.iloc[:,[timeindex,regvarindex]].to_numpy().astype("float64")
    # print(dfmat)
    return(calc_slope_bootstrap(dfmat))

# Loops calculation for several variables
def calc_slope_bootstrap_wrap_loop(df,timeindex,regvarindexes, nsamp = 30):
    nregs = len(regvarindexes)
    outmat = np.zeros([nregs,2])

    for i in np.arange(nregs):
        dfmat = df.iloc[:,[timeindex,regvarindexes[i]]].to_numpy().astype("float64")
        # print((dfmat))
        if np.all(np.isnan(dfmat[:,1])):
            outmat[i,:] = np.nan
        else:
            outmat[i,:] = calc_slope_bootstrap(dfmat, nsamp = nsamp)
        
    return(outmat)

# Calculate linear regression slopes between a variable (timevarname) and several 
# other variables (regvarnames), estimating coefficient variances using bootstrap
def df_apply_reg_bootstrap(indf, timevarname, regvarnames, group_coords=[], nsamp = 30):
    usevarnames = group_coords + [timevarname] + regvarnames

    # Reset indexes
    rindf = indf.reset_index()

    # Get column indexes to use numba
    timeindex = rindf.columns.get_loc(timevarname)
    regvarindexes = [rindf.columns.get_loc(i) for i in regvarnames]

    # Grouped DataFrame, if asked
    #FIXME: Breaks if we don't group, since the .apply used is specifically the GroupBy one
    if len(group_coords) > 0:
        grindf = rindf.groupby(group_coords)
    else:
        grindf = rindf

    # Apply the regressions. This will return a Series of n x 2 arrays, 
    # n being the number of regvars, with mean and variance in the columns
    regseries = grindf.apply(
        functools.partial(calc_slope_bootstrap_wrap_loop, timeindex = timeindex, regvarindexes = regvarindexes, nsamp = nsamp)).rename("slope")

    # Unpack the series into a Dataframe with Series of [mean,variance]
    regdf = pd.concat([regseries.apply(lambda x: x[i,:]).rename(regvarnames[i]) for i in range(len(regvarnames))], axis=1)

    # Unpack each variable in a list of Dataframes with a mean Series and a variance Series with the _var suffix
    dflist = [
        pd.concat([regdf[regvarnames[i]].apply(lambda x: x[0]),
        regdf[regvarnames[i]].apply(lambda x: x[1]).rename(regvarnames[i]+"_var")], axis=1) 
        for i in range(len(regvarnames))
    ]

    # Finally, merge all Dataframes in one
    regvardf = functools.reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), dflist)

    return(regvardf)

# Calculate regressions between each variable in a Dataset
# and the dimension timevarname, estimating the coefficient's 
# variance using bootstrap
def ds_apply_reg_bootstrap(inputds, timevarname, nsamp = 30):
    # If input is a DataArray, make it a Dataset
    if isinstance(inputds, xr.DataArray):
        inputds = inputds.to_dataset()

    regvarnames = list(inputds.data_vars)

    # Calculating one variable at a time and concatenating
    # the results saves a lot of memory on Pandas Dataframe wrangling
    # It also allows Dataarrays with different dimensions
    bigoutds = xr.Dataset()
    for regvarname in regvarnames:
        vards = inputds[[regvarname]]
        orig_coords = list(vards.coords)
        group_coords = [i for i in orig_coords if i != timevarname]
        
        inputdf = vards.to_dataframe()
        bootregdf = df_apply_reg_bootstrap(inputdf, timevarname, [regvarname], group_coords, nsamp = nsamp)
        outds = xr.Dataset.from_dataframe(bootregdf)
        bigoutds = bigoutds.merge(outds)
    
    # inputdf = inputds.to_dataframe()
    # bootregdf = df_apply_reg_bootstrap(inputdf, timevarname, regvarnames, group_coords, nsamp = nsamp)
    # outds = xr.Dataset.from_dataframe(bootregdf)

    return(bigoutds)


# # bigds = xr.open_dataset("tests/poi.nc")
# bigds = xr.open_dataset("tests/trend_test.nc")
# inputds = bigds[["tempmean","agestimate"]].isel(scenario=[1,3], member=0)
# # inputds = bigds[["tempmean","vpdmean"]].isel(scenario=[1,3], member=0)
# # testds = bigds[["tempmean"]]
# # # testds = bigds[["tempmean","vpdmean"]]

# # # # inputds = testds
# timevarname = "year"
# nsamp = 100
# # # outds = ds_apply_reg_bootstrap(inputds, timevarname, nsamp)
# # testoutds = ds_apply_reg_bootstrap(testds, timevarname, nsamp)

# testoutds = ds_apply_reg_bootstrap(inputds, timevarname, nsamp)
# # testoutds = ds_apply_reg_bootstrap(bigds, timevarname, nsamp)

# # %timeit -n 1 -r 1 testoutds = ds_apply_reg_bootstrap(inputds, timevarname, nsamp)
# # # %timeit -n 1 -r 1 testoutds = ds_apply_reg_bootstrap(testds, timevarname, nsamp)
# # # %timeit -n 1 -r 1 testoutds = ds_apply_reg_bootstrap(testds, timevarname, nsamp)

# # # outds["tempmean"].plot(row="scenario")

# testoutds["agestimate"].plot(col="scenario",row = "statmodel")

# # inputds.polyfit(dim = "year", deg =1)["tempmean_polyfit_coefficients"].sel(degree=1).plot(row="scenario")

# # %%
