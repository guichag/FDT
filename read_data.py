"""Read NC files"""

import os
import netCDF4
import numpy as np
import numpy.ma as ma
import cftime

from datetime import datetime

from dataa.d_config import DIRNAME


### CST ###


### FUNC ###

#~Read file

def get_nc_path(ds, source, experiment, member, lat_res, lon_res, var):
    """get nc files path"""
    path = DIRNAME + '/' + ds + '/' + source + '/' + experiment + '/' + member + '/' + str(lat_res) + "x" + str(lon_res) + '/' + var

    return path


def get_nc_file(ds, source, experiment, member, lat_res, lon_res, var, yr):
    """get nc files path"""
    path = get_nc_path(ds, source, experiment, member, lat_res, lon_res, var)
    filename = path + '/' + yr + '.nc'

    return filename


def get_nc_dataset(ds, source, experiment, member, lat_res, lon_res, var, yr):
    """return the nc file dataset"""
    f = get_nc_file(ds, source, experiment, member, lat_res, lon_res, var, yr)
    output = netCDF4.Dataset(f, mode='r')

    return output


#~ Read coords

def get_lat_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lat):
    """Get indices of latitude"""
    data = get_nc_dataset(ds, source, experiment, member, lat_res, lon_res, var, yr)
    lats = data.variables['lat'][:]

    idx = (np.abs(lats - lat)).argmin()

    return idx


def get_lon_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lon):
    """Get indices of longitude"""
    data = get_nc_dataset(ds, source, experiment, member, lat_res, lon_res, var, yr)
    lons = data.variables['lon'][:]

    idx = (np.abs(lons - lon)).argmin()

    return idx


def get_coords(ds, source, experiment, member, lat_res, lon_res, var, yr, lats=(), lons=()):
    """Get coordinates"""
    data = get_nc_dataset(ds, source, experiment, member, lat_res, lon_res, var, yr)

    lats_ = data.variables['lat'][:]
    lons_ = data.variables['lon'][:]

    #for lat, lon in zip(lats, lons):
    #    assert lat in lats_, 'latitude not in dataset'
    #    assert lon in lons_, 'longitude not in dataset'

    if len(lats) == 0:
        lat_s = None
        lat_n = None
    elif len(lats) == 1:
        lat_s = get_lat_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lats[0])
        lat_n = None
    elif len(lats) == 2:
        lat_s = get_lat_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lats[0])
        lat_n = get_lat_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lats[1]) + 1

    if len(lons) == 0:
        lon_w = None
        lon_e = None
    elif len(lons) == 1:
        lon_w = get_lon_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lons[0])
        lon_e = None
    elif len(lons) == 2:
        lon_w = get_lon_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lons[0])
        lon_e = get_lon_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lons[1]) + 1

    out_lats = ma.getdata(lats_[lat_s:lat_n])  
    out_lons = ma.getdata(lons_[lon_w:lon_e])

    return out_lats, out_lons


#~ Read times

def get_times(ds, source, experiment, member, lat_res, lon_res, var, yr):
    """Get times"""
    data = get_nc_dataset(ds, source, experiment, member, lat_res, lon_res, var, yr)
    time = data.variables['time']
    times = netCDF4.num2date(time[:], time.units, time.calendar)
    times = [t.replace(microsecond=0) for t in times]

    return times


def get_time_index(ds, source, experiment, member, lat_res, lon_res, var, yr, m, d):
    data = get_times(ds, source, experiment, member, lat_res, lon_res, var, yr)
    dates = [d.strftime('%Y-%m-%d %H:%M:%S') for d in data]

    yr_ = int(yr)  # delete 0 in yr
    yr_ = str(yr_)

    if ((source == 'CESM2') or (source == 'CESM1-CAM5') or (source == 'GFDL-CM4')) and (len(yr_) < 4): # or (source == 'ACCESS-ESM1-5'): -> FOR 1pctCO2
        h = 0
        dates = [d[4-len(yr_):] for d in dates]
    elif (source == 'CESM2') or (source == 'CESM1-CAM5'):
        h = 0
    else:
        h = 12

    if (source == 'SAM0-UNICON') and (m == 1):
        d = 2

    dates = np.asarray(dates, dtype=str)

    yr_ = int(yr)
    date = datetime(yr_, m, d, h).strftime('%Y-%m-%d %H:%M:%S')

    out_time = np.where(dates == date)[0][0]

    return out_time


def get_time_index_monthly(ds, source, experiment, member, lat_res, lon_res, var, yr, m):
    """Get date index"""
    data = get_times(ds, source, experiment, member, lat_res, lon_res, var, yr)
    #dates = [d.ctime() for d in data]
    dates = np.asarray(data)

    yr_ = int(yr)
    date = datetime(yr_, m, 1)

    out_time = np.where(dates == date)[0][0]

    return out_time


def get_time_index_monthly_by_hour(ds, source, experiment, member, lat_res, lon_res, var, yr, m, h):
    """Get date index"""
    data = get_times(ds, source, experiment, member, lat_res, lon_res, var, yr)
    yr_ = int(yr)
    date = datetime(yr_, m, 1, h)

    out_time = np.where(data == date)[0][0]

    return out_time



