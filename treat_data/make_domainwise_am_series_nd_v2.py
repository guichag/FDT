"""Make AM series from CMIP6 data"""

import sys
import os
import argparse
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr

from global_land_mask import globe

from utils.misc import natural_keys
from dataa.d_config import INDIR, lat_res, lon_res, lats_min, lats_max, lons_min, lons_max
from cmip6.treat_data.t_config import DATADIR
from cmip6.read_data.read_data import get_nc_file


### CST ###


### FUNC ###


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument("--dataset", help="dataset", type=str, default='CMIP6')
    parser.add_argument("--source", help="source (GCM)", type=str, default='IPSL-CM6A-LR')
    parser.add_argument("--experiment", help="experiment (historical)", type=str, default='historical')
    parser.add_argument("--member", nargs="+", help="member", type=str, default=None)
    parser.add_argument("--variable", help="variable to consider", type=str, default='pr')
    parser.add_argument("--start_month", help="period start month", type=int, default=1)
    parser.add_argument("--start_day", help="period start day", type=int, default=1)
    parser.add_argument("--end_month", help="period end month", type=int, default=12)
    parser.add_argument("--end_day", help="period end day", type=int, default=31)
    parser.add_argument("--ymin", help="start year", type=int, default=1950)
    parser.add_argument("--ymax", help="end year", type=int, default=2014)
    parser.add_argument("--ndays", help="RXnD", type=int, default=7)

    opts = parser.parse_args()

    ds = opts.dataset
    src = opts.source
    exp = opts.experiment
    mem = opts.member
    var = opts.variable
    bmonth = opts.start_month
    bday = opts.start_day
    emonth = opts.end_month
    eday = opts.end_day
    ymin = opts.ymin
    ymax = opts.ymax
    nd = opts.ndays

    years = np.arange(ymin, ymax+1, 1)
    years_ = [str(y) for y in years]

    bdate = [bmonth, bday]
    edate = [emonth, eday]

    if mem == None:
        data_path = INDIR + '/' + ds + '/' + src + '/' + exp
        mems = os.listdir(data_path)
    else:
        mems = mem

    mems.sort(key=natural_keys)

    lat_min = -90. # lats_min[ds]
    lat_max = 90. #lats_max[ds]
    lon_min = -180. #lons_min[ds]
    lon_max = 180. # lons_max[ds]

    lat_res_ = lat_res[ds][exp][src]
    lon_res_ = lon_res[ds][exp][src]

    res_ = str(lat_res_) + "x" + str(lon_res_)

    print('Experiment: {0}'.format(exp))
    print('Domain: lat ({0},{1}), lon ({2},{3})'.format(lat_min, lat_max, lon_min, lon_max))
    print('GCM: {0}'.format(src))
    print('Members: {0}'.format(mems))
    print('Resolution: {0}'.format(res_))
    print('RX{0}D'.format(nd))


    #~ Outdir

    if not os.path.isdir(DATADIR + '/am_series'):
        os.mkdir(DATADIR + '/am_series')
    outdir = DATADIR + '/am_series'

    if not os.path.isdir(outdir + '/domain_wise'):
        os.mkdir(outdir + '/domain_wise')
    outdir = outdir + '/domain_wise'

    if not os.path.isdir(outdir + '/' + ds):
        os.mkdir(outdir + '/' + ds)
    outdir = outdir + '/' + ds

    if not os.path.isdir(outdir + '/' + src):
        os.mkdir(outdir + '/' + src)
    outdir = outdir + '/' + src

    if not os.path.isdir(outdir + '/' + exp):
        os.mkdir(outdir + '/' + exp)
    outdir = outdir + '/' + exp


    #~ Get data

    for mem_ in mems:
        print('\n>>> {0} <<<'.format(mem_))

        out_maxs = []
        #out_date_maxs = []

        for y in years_:
            print(y, end=' : ', flush=True)

            ncfile = get_nc_file(ds, src, exp, mem_, lat_res_, lon_res_, 'pr', y)
            
            data = xr.open_dataset(ncfile)
            data = data.pr
            data = data.where(data >= 0)

            data_rm = data.rolling(time=nd).sum()
            data_max = data_rm.max(dim='time')

            lats, lons = np.meshgrid(data_max.lat.values, data_max.lon.values)

            mask_ocean = globe.is_ocean(lats, lons)   # Ocean mask: True (masked) if point is ocean
            mask_ocean = mask_ocean.swapaxes(0,1)

            data_max_land = data_max.where(~mask_ocean)

            out_maxs.append(data_max_land)

        out_maxs_all = xr.concat(out_maxs, dim='time', coords='minimal')
        out_maxs_all = out_maxs_all.assign_coords(time=years)


        #~ Save

        if not os.path.isdir(outdir + '/' + mem_):
            os.mkdir(outdir + '/' + mem_)
        outdir_ = outdir + '/' + mem_

        if not os.path.isdir(outdir_ + '/' + res_):
            os.mkdir(outdir_ + '/' + res_)
        outdir_ = outdir_ + '/' + res_

        if not os.path.isdir(outdir_ + '/lat({0},{1})'.format(lat_min, lat_max) + '_lon({0},{1})'.format(lon_min, lon_max)):
            os.mkdir(outdir_ + '/lat({0},{1})'.format(lat_min, lat_max) + '_lon({0},{1})'.format(lon_min, lon_max))
        outdir_ = outdir_ + '/lat({0},{1})'.format(lat_min, lat_max) + '_lon({0},{1})'.format(lon_min, lon_max)

        outfile = outdir_ + '/amax_' + str(ymin) + '-' + str(ymax) + '_' + str(nd) + 'd'

        out_maxs_all.to_netcdf(outfile)


print('Done')


