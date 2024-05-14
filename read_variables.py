"""Read variables"""

import os
import netCDF4
import numpy as np

from cmip6.read_data.read_data import *


### CST ###


### FUNC ###

def get_var_data(ds='CMIP6', source='IPSL-CM6A-LR', experiment='historical', member='r1i1p1f1', lat_res=1.27, lon_res=2.5, var='pr', yr='1980', time=([1,1], [12,31]), lats=(), lons=()):
    """Get 2D variable data"""
    data = get_nc_dataset(ds, source, experiment, member, lat_res, lon_res, var, yr)
    data = data.variables[var]

    t_b = get_time_index(ds=ds, source=source, experiment=experiment, member=member, lat_res=lat_res, lon_res=lon_res, var=var, yr=yr, m=time[0][0], d=time[0][1])
    t_e = get_time_index(ds=ds, source=source, experiment=experiment, member=member, lat_res=lat_res, lon_res=lon_res, var=var, yr=yr, m=time[1][0], d=time[1][1]) + 1

    coords = get_coords(ds=ds, source=source, experiment=experiment, member=member, lat_res=lat_res, lon_res=lon_res, var=var, yr=yr)

    lats_ = coords[0]
    lons_ = coords[1]


    if len(lats) == 0:
        lat_s = None
        lat_n = None
    elif len(lats) == 1:
        lat_s = get_lat_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lats[0]) # + 1
        lat_n = None # get_lat_index(ds, var, lat_res, lon_res, yr, lats[0])
    elif len(lats) == 2:
        lat_s = get_lat_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lats[0]) # + 1
        lat_n = get_lat_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lats[1]) + 1

    if len(lons) == 0:
        lon_w = None
        lon_e = None
    elif len(lons) == 1:
        lon_w = get_lon_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lons[0])
        lon_e = None # get_lon_index(ds, var, lat_res, lon_res, yr, lons[0]) + 1
    elif len(lons) == 2:
        lon_w = get_lon_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lons[0])
        lon_e = get_lon_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lons[1]) + 1

    output = data[t_b:t_e, lat_s:lat_n, lon_w:lon_e]

    return output


    ### NEXT: TO UPDATE ###


def get_3dvar_data(ds, var, lat_res, lon_res, yr, time=([7, 1, 0], [9, 30, 18]), lvl=(), lats=(), lons=()):
    """Get 3D variable data"""
    data = get_nc_dataset(ds, var, lat_res, lon_res, yr)

    t_b = get_time_index_hourly(ds, var, lat_res, lon_res, yr, time[0][0], time[0][1], time[0][2])
    t_e = get_time_index_hourly(ds, var, lat_res, lon_res, yr, time[1][0], time[1][1], time[1][2]) + 1

    if len(lvl) == 0:
        lvl_0 = None
        lvl_1 = None
    elif len(lvl) == 1:
        lvl_0 = lvl[0]
        lvl_1 = lvl[0]+1
    elif len(lvl) == 2:
        lvl_0 = lvl[0]
        lvl_1 = lvl[1]

    if len(lats) == 0:
        lat_s = None
        lat_n = None
    elif len(lats) == 1:
        lat_s = get_lat_index(ds, var, lat_res, lon_res, yr, lats[0]) # + 1
        lat_n = None # get_lat_index(ds, var, lat_res, lon_res, yr, lats[0])
    elif len(lats) == 2:
        lat_s = get_lat_index(ds, var, lat_res, lon_res, yr, lats[0]) + 1
        lat_n = get_lat_index(ds, var, lat_res, lon_res, yr, lats[1])

    if len(lons) == 0:
        lon_w = None
        lon_e = None
    elif len(lons) == 1:
        lon_w = get_lon_index(ds, var, lat_res, lon_res, yr, lons[0])
        lon_e = None # get_lon_index(ds, var, lat_res, lon_res, yr, lons[0]) + 1
    elif len(lons) == 2:
        lon_w = get_lon_index(ds, var, lat_res, lon_res, yr, lons[0])
        lon_e = get_lon_index(ds, var, lat_res, lon_res, yr, lons[1]) # + 1

    output = data.variables[var][t_b:t_e, lvl_0:lvl_1, lat_n:lat_s, lon_w:lon_e]

    return output


def get_var_data_monthly(ds='CMIP6', source='IPSL-CM6A-LR', experiment='historical', member='r1i1p1f1', lat_res=1.27, lon_res=2.5, var='tas', yr='1980', time=(1, 12), lats=(), lons=()):
    """Get data"""
    data = get_nc_dataset(ds, source, experiment, member, lat_res, lon_res, var, yr)
    data = data.variables[var]

    t_b = time[0] - 1 # get_time_index_monthly(ds=ds, source=source, experiment=experiment, member=member, lat_res=lat_res, lon_res=lon_res, var=var, yr=yr, m=time[0])
    t_e = time[1] # get_time_index_monthly(ds=ds, source=source, experiment=experiment, member=member, lat_res=lat_res, lon_res=lon_res, var=var, yr=yr, m=time[1]) + 1

    coords = get_coords(ds=ds, source=source, experiment=experiment, member=member, lat_res=lat_res, lon_res=lon_res, var=var, yr=yr)

    lats_ = coords[0]
    lons_ = coords[1]


    if len(lats) == 0:
        lat_s = None
        lat_n = None
    elif len(lats) == 1:
        lat_s = get_lat_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lats[0]) # + 1
        lat_n = None # get_lat_index(ds, var, lat_res, lon_res, yr, lats[0])
    elif len(lats) == 2:
        lat_s = get_lat_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lats[0]) # + 1
        lat_n = get_lat_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lats[1]) + 1

    if len(lons) == 0:
        lon_w = None
        lon_e = None
    elif len(lons) == 1:
        lon_w = get_lon_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lons[0])
        lon_e = None # get_lon_index(ds, var, lat_res, lon_res, yr, lons[0]) + 1
    elif len(lons) == 2:
        lon_w = get_lon_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lons[0])
        lon_e = get_lon_index(ds, source, experiment, member, lat_res, lon_res, var, yr, lons[1]) + 1

    output = data[t_b:t_e, lat_s:lat_n, lon_w:lon_e]

    return output


def get_var_data_monthly_by_hour(ds, var, lat_res, lon_res, yr, time=(1, 12), h=18, lats=(), lons=()):
    """Get data"""
    data = get_nc_dataset(ds, var, yr)

    t_b = get_time_index_monthly_by_hour(ds, var, lat_res, lon_res, yr, time[0], h)
    t_e = get_time_index_monthly_by_hour(ds, var, lat_res, lon_res, yr, time[1], h) + 1

    if len(lats) == 0:
        lat_s = None
        lat_n = None
    elif len(lats) == 1:
        lat_s = get_lat_index(ds, var, lat_res, lon_res, yr, lats[0]) #+ 1
        lat_n = None
    elif len(lats) == 2:
        lat_s = get_lat_index(ds, var, lat_res, lon_res, yr, lats[0]) #+ 1
        lat_n = get_lat_index(ds, var, lat_res, lon_res, yr, lats[1])

    if len(lons) == 0:
        lon_w = None
        lon_e = None
    elif len(lons) == 1:
        lon_w = get_lon_index(ds, var, lat_res, lon_res, yr, lons[0]) #+ 1
        lon_e = None
    elif len(lons) == 2:
        lon_w = get_lon_index(ds, var, lat_res, lon_res, yr, lons[0])
        lon_e = get_lon_index(ds, var, lat_res, lon_res, yr, lons[1]) #+ 1

    output = data.variables[var][t_b:t_e, lat_n:lat_s, lon_w:lon_e]

    return output




