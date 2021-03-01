#%%
import numpy as np
import xarray as xr
from scipy.stats import ttest_ind_from_stats
import re
import numba
from numba import jit, vectorize, guvectorize
import dask.array as da

# This is basically a weighted mean that assumes 
# a has weight rn and b has weight 1
# @jit(nopython = True)
@vectorize(['float64(float64,float64,float64)'], nopython = True)
def pool_pair_means(a,b,rn):
    out = (rn*a + b)/(rn+1)
    return(out)

# This is just a fancy way to calculate a grand
# mean by "accumulating" means. The function itself
# is not particularly useful since it does the same as
# np.mean, but the logic can be handy if we are accumulating
# means without knowing the final n 
@jit(nopython = True)
# @vectorize(['float64(float64)'])
# @vectorize(['float64(float64)'], nopython = True)
def pool_means(v):
    # i = 0
    s = v[0]
    # while i < (len(v)-1):
    for i in range(v.shape[0]-1):
        rn = i + 1
        s = pool_pair_means(s,v[i+1],rn)
        i = i + 1
    return(s)

# Get the pooled variance of two subsamples of the same
# distribution given just their means, variances and numbers of observations
# Assumes all subsamples were independent from each other and sampled from 
# the same distribution
# @jit(nopython = True)
@vectorize(['float64(float64, float64, float64, float64, float64, float64)'], nopython = True)
def pool_pair_variances(v1,v2,m1,m2,n1,n2):
    vp = (1/(n1+n2-1)) * ( (n1-1)*v1 + (n2-1)*v2 + ((n1*n2)/(n1+n2)) * (m1-m2)**2 )
    return(vp)

# This is a simple ufunc (not generalize), it doesn't vectorize well
# WARNING: Do not use
# Get the pooled variance of a series of subsamples of the same
# distribution given just their means, variances and the number
# of observations of each subsample, assuming they are all the 
# same and equal to n.
# Assumes all subsamples were independent from each other and sampled from 
# the same distribution
@jit(nopython = True)
# @vectorize(['float64(float64, float64, float64)'])
def pool_variances(means, variances,n):
    i = 0
    vp = variances[0]
    mp = means[0]
    while i < (variances.shape[0]-1):
        rn = i + 1
        vp = pool_pair_variances(vp, variances[i+1], mp, means[i+1], n*rn, n)
        mp = pool_pair_means(mp,means[i+1],rn)
        i = i + 1
    return(vp)

# This is a dask-enabled GUfunc version of pool_variances
# Get the pooled variance of a series of subsamples of the same
# distribution given just their means, variances and the number
# of observations of each subsample, assuming they are all the 
# same and equal to n.
# Assumes all subsamples were independent from each other and sampled from 
# the same distribution
# @guvectorize("(float64[:],float64[:],float64,float64)", "(m),(m),()->()", nopython = True) # This has some weird bug where the loop retains the last element of the input instead of actually accumulating input
# ====== WORKING
@da.as_gufunc(signature="(m),(m),()->()", output_dtypes=float, vectorize=True)
@jit(nopython = True)
# ========
def pool_variances_dask(means, variances,n):
    i = 0
    vp = variances[0]
    mp = means[0]
    for i in range(variances.shape[0]-1):
        rn = i + 1
        # vp = pool_pair_variances_dask(vp, variances[i+1], mp, means[i+1], n*rn, n)
        vp = pool_pair_variances(vp, variances[i+1], mp, means[i+1], n*rn, n)
        # mp = pool_pair_means_dask(mp,means[i+1],rn)
        mp = pool_pair_means(mp,means[i+1],rn)
        # i = i + 1
    return(vp)

# Add a suffix to all variables in a Dataset
def add_suf(ds,suf):
    return(ds.rename({i:(i + suf) for i in ds.data_vars}))

# Removes a suffix from all variables in a Dataset
def rem_suf(ds,suf):
    return(ds.rename({i:(re.sub("(.*)"+suf+"$","\g<1>",i)) for i in ds.data_vars}))

# Splits a dataset that has means and variances coded by name. 
# e.g. the mean of variable X is in variable X and it's variance is in X_val
def split_dataset_variances_generic(dsboth):
    regexvar = re.compile(".*_var$")
    dsmeans = dsboth[[i for i in dsboth.data_vars if not regexvar.match(i)]]
    dsvariances = dsboth[[i for i in dsboth.data_vars if regexvar.match(i)]]
    dsvariances = dsvariances.rename({i:(re.sub("(.*)_var$","\g<1>",i)) for i in dsvariances.data_vars})

    # Select just the desired variables
    # dsmeans = dsmeans[selvars]
    # dsvariances = dsvariances[selvars]

    return((dsmeans, dsvariances))

# Does the opposite, combining two datasets by appending "_var" to one of them
def combine_dataset_variances_generic(dsmeans, dsvariances):
    dsvariances = add_suf(dsvariances,"_var")
    dsout = dsmeans.merge(dsvariances)
    return(dsout)

# Adds or subtracts two datasets and combine their variances,
# assuming they are uncorrelated. Therefore the variances are just added
def addsub_ds_variances(ds1,ds2,addsub):
    (ds1means, ds1variances) = split_dataset_variances_generic(ds1)
    (ds2means, ds2variances) = split_dataset_variances_generic(ds2)

    if addsub == "add":
        dsresmeans = ds1means + ds2means
    elif addsub == "sub":
        dsresmeans = ds1means - ds2means
    else:
        raise NameError("addsub_ds_variances() argument addsub must be either 'add' or 'sub'")
    dsresvariances = ds1variances + ds2variances

    dsout = combine_dataset_variances_generic(dsresmeans, dsresvariances)
    return(dsout)
    

# Calculates the difference between "contvarname" and "overvars" in a dataset
# and also returns a "contvarname"_pval variable with the p-values of a
# t-test on the difference of two means given a "contvarname"_var variable
def calc_diff_ttest_generic(dsboth1, dsboth2, nobs):

    (dsmeans1, dsvariances1) = split_dataset_variances_generic(dsboth1)
    (dsmeans2, dsvariances2) = split_dataset_variances_generic(dsboth2)

    diff = dsmeans1 - dsmeans2

    # t-test
    # testvarnames = selvars

    dsttest = xr.apply_ufunc(
            ttest_ind_from_stats,
            dsmeans1,
            dsvariances1**0.5,
            nobs,
            dsmeans2,
            dsvariances2**0.5,
            nobs,
            True,
            input_core_dims=[[], [], [], [], [], [], []],
            output_core_dims=[[], []],
            vectorize=True,
            # keep_attrs=True,
            dask='parallelized',
        )[1]

    dsttest = dsttest.rename({i:(i + "_pval") for i in dsttest.data_vars})
    
    # Variance of the difference
    dsvariances = dsvariances1 + dsvariances2
    dsvariances = dsvariances.rename({i:(i + "_var") for i in dsvariances.data_vars})
    
    diff = diff.merge(dsttest)
    diff = diff.merge(dsvariances)
    # diff.expand_dims("lev")
    # diff["lev"] = np.array(uselev)
    # (diff,dump) = xr.broadcast(diff, dsboth1)
    return(diff)

# #%%
# mat = np.random.normal(4,1,size = (2,3,4,40))
# mmat = np.mean(mat, axis = 3)
# vmat = np.var(mat,ddof=1, axis = 3)
# compvmat = np.var(mat,ddof=1, axis = (2,3))

# # pool_pair_variances_dask(vmat[0,:], vmat[1,:], mmat[0,:], mmat[1,:], 1000, 10).compute()
# pool_pair_variances(vmat[0,0], vmat[1,0], mmat[0,0], mmat[1,0], 1000, 10)
# # pool_pair_variances_dask(vmat[0,0], vmat[1,0], mmat[0,0], mmat[1,0], 1000, 10).compute()

# # pool_variances_dask(da.array(mmat), da.array(vmat), 40.0).compute()
# out = pool_variances_dask(mmat, vmat, 40.0).compute()
# # out = pool_variances_dask(mmat, vmat, 40.0)

# print(compvmat)
# print(out)

# #%%
# import xarray as xr
# import re

# dsin = xr.open_dataset("temp_ds.nc")

# mvarname = "PSL"
# vvarname = mvarname + "_var"


# # pool_variances_dask(dsin["TS"], dsin["TS_var"], 10.0).compute()
# # out = da.apply_gufunc(pool_variances_dask, pool_variances_dask.signature, dsin[mvarname], dsin[vvarname], 10.0, \
#     # allow_rechunk = True, axis = 0)

# # out = xr.apply_ufunc(pool_variances_dask, dsin[mvarname], dsin[vvarname], 10.0, \
# #     dask = "parallelized", input_core_dims=[["member"], ["member"], [] ],
# #     keep_attrs=True)

# # dstest = dsin[[i for i in dsin.data_vars if i not in ["T","T_var"]]]
# dstest = dsin

# regexvar = re.compile(".*_var$")
# dsmeans = dstest[[i for i in dstest.data_vars if not regexvar.match(i)]]
# dsvars = dstest[[i for i in dstest.data_vars if regexvar.match(i)]]
# dsvars = dsvars.rename({i:(re.sub("(.*)_var$","\g<1>",i)) for i in dsvars.data_vars})

# out = xr.apply_ufunc(pool_variances_dask, dsmeans, dsvars, 10.0, \
#     dask = "parallelized", input_core_dims=[["member"], ["member"], [] ],
#     keep_attrs=True)

# out
# # out.sel(month = 1).plot()
# out.to_netcdf("apagar.nc")


#%%
# ============== UNUSED
# These were dask versions of the pairwise functions
# They don't work well inside the loop because the (i)
# signature apparently does not work well with scalars

# # This is basically a weighted mean that assumes 
# # a has weight rn and b has weight 1
# # @jit(nopython = True)
# # @vectorize(['float64(float64,float64,float64)'])
# @da.as_gufunc(signature="(i),(i)->(i)", output_dtypes=float, vectorize=True)
# def pool_pair_means_dask(a,b,rn):
#     out = (rn*a + b)/(rn+1)
#     return(out)

# # Get the pooled variance of two subsamples of the same
# # distribution given just their means, variances and numbers of observations
# # Assumes all subsamples were independent from each other and sampled from 
# # the same distribution
# # @jit(nopython = True)
# # @vectorize(['float64(float64, float64, float64, float64, float64, float64)'], nopython = True)
# @da.as_gufunc(signature="(i),(i),(i),(i),(),()->(i)", output_dtypes=float, vectorize=True)
# def pool_pair_variances_dask(v1,v2,m1,m2,n1,n2):
#     vp = (1/(n1+n2-1)) * ( (n1-1)*v1 + (n2-1)*v2 + ((n1*n2)/(n1+n2)) * (m1-m2)**2 )
#     return(vp)

#%%
# import inspect
# import sys

# def recompile_nb_code():
#     this_module = sys.modules[__name__]
#     module_members = inspect.getmembers(this_module)

#     for member_name, member in module_members:
#         if hasattr(member, 'recompile') and hasattr(member, 'inspect_llvm'):
#             member.recompile()
# recompile_nb_code()

# # del(soma)
# @guvectorize(["float64[:], float64[:]"], "(m)->()", nopython = True, target = "cpu")
# def soma(a, out):
#     m = a.shape[0]
#     out = 0.0
#     for i in range(a.shape[0]):
#         # out = out + a[i] #+ b[i]
#         # out = out + a[i] 
#         out += a[i]
#     # print(out)
#     # out = 30.0
#     # out = a
#     # return(out)

# a = np.array([0.,1,2,3])
# b = np.array([10.,11,12,13])

# soma(a)

#%%
# sys.path.append("fortran/")
# import dpres
# dpres.dpres.dpres1d()