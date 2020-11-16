#%%
import numpy as np
import numba
from numba import jit, vectorize, guvectorize
import dask.array as da

# This is basically a weighted mean that assumes 
# a has weight rn and b has weight 1
# @jit(nopython = True)
@vectorize(['float64(float64,float64,float64)'])
def pool_pair_means(a,b,rn):
    out = (rn*a + b)/(rn+1)
    return(out)

# This is just a fancy way to calculate a grand
# mean by "accumulating" means. The function itself
# is not particularly useful since it does the same as
# np.mean, but the logic can be handy if we are accumulating
# means without knowing the final n 
# @jit(nopython = True)
@vectorize(['float64(float64)'])
def pool_means(v):
    i = 0
    s = v[0]
    while i < (len(v)-1):
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

# Get the pooled variance of a series of subsamples of the same
# distribution given just their means, variances and the number
# of observations of each subsample, assuming they are all the 
# same and equal to n.
# Assumes all subsamples were independent from each other and sampled from 
# the same distribution
# @jit(nopython = True)
@vectorize(['float64(float64, float64, float64)'])
def pool_variances(means, variances,n):
    i = 0
    vp = variances[0]
    mp = means[0]
    while i < (len(variances)-1):
        rn = i + 1
        vp = pool_pair_variances(vp, variances[i+1], mp, means[i+1], n*rn, n)
        mp = pool_pair_means(mp,means[i+1],rn)
        i = i + 1
    return(vp)

#%%
# # Some tests
bigv = np.random.normal(4,1,size = 40)
np.var(bigv, axis=1)
n = 10
subv = [bigv[i:i+n] for i in range(0,len(bigv),n)]
# means = [np.mean(i) for i in subv]
# variances = [np.var(i, ddof = 1) for i in subv]

# pool_variances(means,variances,n)
# np.var(bigv, ddof = 1)

# pool_pair_variances(variances[0],variances[1],means[0],means[1],len(subv[0]),len(subv[1]))
# np.var(np.concatenate([subv[0],subv[1]]), ddof =1 )

# pool_means(means)

#%%
import xarray as xr

# squared_error = lambda x, y: (x - y) ** 2
# arr1 = xr.DataArray([0, 1, 2, 3], dims="x")
# xr.apply_ufunc(squared_error, arr1, 1)

mat = np.random.normal(4,1,size = (2,4,40))
mmat = np.mean(mat, axis = 2)
vmat = np.var(mat,ddof=1, axis = 2)
compvmat = np.var(mat,ddof=1, axis = (1,2))

pool_means.reduce(mmat, axis = 1)

arr = xr.DataArray([0, 1, 2, 3], dims="x")



myfunc = lambda x: np.sum(x)

xr.apply_ufunc(myfunc, arr)

myfunc(arr.data)


mmat
# %%
# mmat

@guvectorize(['float64[:], float64[:]'], '(n)->()', nopython = True)
def soma(inmat, outmat):
    outmat = 10
    for i in range(inmat.shape[0]):
        outmat = outmat + inmat[i]

inmat = np.arange(5, dtype='float64')
# outmat = np.zeros_like(inmat)
outmat = np.array(0, dtype='float64')
soma(inmat, outmat)
outmat

# %%

# @guvectorize(['float64[:], float64, float64[:]'], '(n),()->(n)')
@guvectorize(['float64[:], float64, float64'], '(n),()->()')
def g(x, y, res):
    res = 1.0
    for i in range(x.shape[0]):
        res = res + x[i] - y
        print(res)



a = np.arange(6)*2.0

# a = np.arange(6).reshape(2, 3)

# out = np.array(0.0)
g(a,30)

# x=a
# y=10
# res = 2.0
# for i in range(x.shape[0]):
#     res = res + x[i] + y
#     print(res)
# res

# %%
x = da.random.normal(size=(10, 5), chunks=(2, 5))

@da.as_gufunc(signature="(i)->()", output_dtypes=float, vectorize=True)
def gufoo(x):
    return np.mean(x, axis=-1)

y = gufoo(x)
#%%
a = da.random.normal(size=(10, 5), chunks=(2, 5))
b = da.random.normal(size=(10, 5), chunks=(2, 5))

a3 = da.random.normal(size=(10, 5, 2), chunks=(2, 5, 2))
b3 = da.random.normal(size=(10, 5, 2), chunks=(2, 5, 2))


@da.as_gufunc(signature="(i),(i)->()", output_dtypes=float, vectorize=True)
def gusum(x,y):
    # return np.mean(x+y, axis=-1)
    s = 0.0
    for i in range(x.shape[0]):
        s = s + x[i] + y[i]
    return(s)

# y = gusum(a,b)
y = gusum(a3,b3)
y.compute()

#%%

# This is basically a weighted mean that assumes 
# a has weight rn and b has weight 1
# @jit(nopython = True)
# @vectorize(['float64(float64,float64,float64)'])
@da.as_gufunc(signature="(i),(i)->(i)", output_dtypes=float, vectorize=True)
def pool_pair_means_dask(a,b,rn):
    out = (rn*a + b)/(rn+1)
    return(out)

# Get the pooled variance of two subsamples of the same
# distribution given just their means, variances and numbers of observations
# Assumes all subsamples were independent from each other and sampled from 
# the same distribution
# @jit(nopython = True)
# @vectorize(['float64(float64, float64, float64, float64, float64, float64)'], nopython = True)
@da.as_gufunc(signature="(i),(i),(i),(i),(),()->(i)", output_dtypes=float, vectorize=True)
def pool_pair_variances_dask(v1,v2,m1,m2,n1,n2):
    vp = (1/(n1+n2-1)) * ( (n1-1)*v1 + (n2-1)*v2 + ((n1*n2)/(n1+n2)) * (m1-m2)**2 )
    return(vp)

# Get the pooled variance of a series of subsamples of the same
# distribution given just their means, variances and the number
# of observations of each subsample, assuming they are all the 
# same and equal to n.
# Assumes all subsamples were independent from each other and sampled from 
# the same distribution
# @jit(nopython = True)
# @vectorize(['float64(float64, float64, float64)'])
@da.as_gufunc(signature="(m),(m),()->()", output_dtypes=float, vectorize=True)
def pool_variances_dask(means, variances,n):
    i = 0
    vp = variances[0]
    mp = means[0]
    while i < (variances.shape[0]-1):
        rn = i + 1
        # vp = pool_pair_variances_dask(vp, variances[i+1], mp, means[i+1], n*rn, n)
        vp = pool_pair_variances(vp, variances[i+1], mp, means[i+1], n*rn, n)
        # mp = pool_pair_means_dask(mp,means[i+1],rn)
        mp = pool_pair_means(mp,means[i+1],rn)
        i = i + 1
    return(vp)


mat = np.random.normal(4,1,size = (2,4,40))
mmat = np.mean(mat, axis = 2)
vmat = np.var(mat,ddof=1, axis = 2)
compvmat = np.var(mat,ddof=1, axis = (1,2))

# pool_pair_variances_dask(vmat[0,:], vmat[1,:], mmat[0,:], mmat[1,:], 1000, 10).compute()
pool_pair_variances(vmat[0,0], vmat[1,0], mmat[0,0], mmat[1,0], 1000, 10)
# pool_pair_variances_dask(vmat[0,0], vmat[1,0], mmat[0,0], mmat[1,0], 1000, 10).compute()

pool_variances_dask(da.array(mmat), da.array(vmat), 40.0).compute()

# pool_variances_dask(da.array(mmat[0,:]), da.array(vmat[0,:]), 40.0).compute()


