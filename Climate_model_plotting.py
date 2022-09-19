#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import dask
import numpy as np
import cartopy.feature as cfeature
import scipy.stats as st

from matplotlib import pyplot as plt
from datetime import datetime
from netCDF4 import Dataset
from scipy import stats


# In[ ]:


# Open .nc data files
data = xr.open_dataset('filename.nc', chunks = 'auto')


# # T-test

# In[ ]:


# Enter the desired start/end point of latitude or longitude, and 
def coordinate(start,end,interval):
    if start < 0:
        start = 360 - abs(start)
    if end < 0:
        end = 360 - abs(end)
    interval = abs((start - end)/interval - 1)
    return np.linspace(int(start), int(end), int(interval)) 

