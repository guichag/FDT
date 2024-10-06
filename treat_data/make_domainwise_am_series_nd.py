"""Make AM series from CMIP6 data"""

import sys
import os
import argparse
import dill
import pickle
import numpy as np
import numpy.ma as ma
import pandas as pd

from functions import title_text

from utils.misc import natural_keys
from dataa.d_config import DIRNAME, lat_res, lon_res, lats_min, lats_max, lons_min, lons_max
from cmip6.read_data.read_data import get_times, get_coords, get_lat_index, get_lon_index
from cmip6.read_data.read_variables import get_var_data
from cmip6.treat_data.t_config import DATADIR
from cmip6.treat_data.get_land_points import load_land_points_cmip


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
    years = [str(y) for y in years]

    dates = pd.date_range(start=years[0]+'-01-01', end=years[-1]+'-12-31', freq='D')  # for year in years]
    dates = dates[(dates.month !=2) | (dates.day != 29)]

    bdate = [bmonth, bday]
    edate = [emonth, eday]

    if mem == None:
        data_path = DIRNAME + '/' + ds + '/' + src + '/' + exp
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

    # N-day rolling mean: see https://danielmuellerkomorowska.com/2020/06/02/smoothing-data-by-rolling-average-with-numpy/
    kernel_size = nd
    kernel = np.ones(kernel_size) / kernel_size

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

    land = load_land_points_cmip(ds=ds, source=src, lat_res=lat_res_, lon_res=lon_res_, var=var, lat_sub=(lat_min, lat_max), lon_sub=(lon_min, lon_max))

    for mem_ in mems:
        print(title_text(mem_, 1))

        out_rrs = []

        for y in years:
            print(y, end=' : ', flush=True)

            ly = len(y)

            if (src == 'CESM2') or (src == 'ACCESS-ESM1-5') or (src == 'GFDL-CM4') and (ly < 4):
                y = (4 - ly) * '0' + y

            rr = get_var_data(ds=ds, source=src, experiment=exp, member=mem_, lat_res=lat_res_, lon_res=lon_res_, var=var, yr=y, time=(bdate, edate))
            rr_ = ma.getdata(rr)

            rr_[rr_ < 0.] = np.nan    # th_wd  -> use of WD threshold exclude too many values in north for SSPs

            out_rrs.append(rr_)

        out_rrs = np.asarray(out_rrs)

        coords = get_coords(ds=ds, source=src, experiment=exp, member=mem_, lat_res=lat_res_, lon_res=lon_res_, var=var, yr=y)  #, lats=lat_sub, lons=lon_sub)

        lats = coords[0]
        lons = coords[1]


        out_ams = {}

        for ilat, lat in enumerate(lats):
            for ilon, lon in enumerate(lons):
                coord = (lat, lon)

                land_ = land[ilat, ilon]

                if land_ == False:
                    print("Compute AM series for lat: {0} / lon: {1}".format(lat, lon))

                    rr_ = [pd.Series(out_rr[:,ilat,ilon], index=dates[dates.year == int(years[iout_rr])]) for iout_rr, out_rr in enumerate(out_rrs)]
                    rrs_ = pd.concat(rr_, axis=0)
                    rrs_nd = rrs_.rolling('{0}D'.format(nd)).sum()
                    rrs_nd = rrs_nd.to_frame()
                    rrs_nd_ = rrs_nd.set_index([rrs_nd.index.year, rrs_nd.index.day])
                    rrs_nd_max = rrs_nd_.max(level=0)

                    df_am = pd.Series(data=rrs_nd_max.values.flatten(), index=years)
                    df_am = df_am.dropna()

                    out_ams[coord] = df_am


        #~ Save

        if out_ams != {}:

            print('\n-- Save --')

            if not os.path.isdir(outdir + '/' + mem_):
                os.mkdir(outdir + '/' + mem_)
            outdir_ = outdir + '/' + mem_

            if not os.path.isdir(outdir_ + '/' + res_):
                os.mkdir(outdir_ + '/' + res_)
            outdir_ = outdir_ + '/' + res_

            if not os.path.isdir(outdir_ + '/lat({0},{1})'.format(lat_min, lat_max) + '_lon({0},{1})'.format(lon_min, lon_max)):
                os.mkdir(outdir_ + '/lat({0},{1})'.format(lat_min, lat_max) + '_lon({0},{1})'.format(lon_min, lon_max))
            outdir_ = outdir_ + '/lat({0},{1})'.format(lat_min, lat_max) + '_lon({0},{1})'.format(lon_min, lon_max)


            outfile = outdir_ + '/amax_' + str(nd) + 'd'
            #outfile_date = outdir + '/' + ds + '_date'

            with open(outfile, 'wb') as pics:
                pickle.dump(obj=out_ams, file=pics)

            """with open(outfile_date, 'wb') as pics:
               pickle.dump(file=pics, obj=out_ams_date)"""


    print('Done')


