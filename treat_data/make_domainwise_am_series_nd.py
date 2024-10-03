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
#from cmip6.cmip6_config import TH_WD
from cmip6.read_data.read_data import get_times, get_coords, get_lat_index, get_lon_index
from cmip6.read_data.read_variables import get_var_data
from cmip6.treat_data.t_config import DATADIR
from cmip6.treat_data.get_land_points import load_land_points_cmip


### CST ###


### FUNC ###

def load_am_series_domain(ds='CMIP6', source='IPSL-CM6A-LR', experiment='historical', member='r1i1p1f1', lat_res=1.27, lon_res=2.5, lat_sub=(0., 20.), lon_sub=(-20., 20.)): # None
    """Load AM series for a given domain"""
    res_ = str(lat_res) + "x" + str(lon_res)

    lat_min = -90. # lats_min[ds]
    lat_max = 90. # lats_max[ds]
    lon_min = -180. # lons_min[ds]
    lon_max = 180. # lons_max[ds]

    if lat_sub:
        assert lat_sub[0] <= lat_sub[1], 'wrong latitude order'
        lat_min_ = lat_sub[0]
        lat_max_ = lat_sub[1]
    else:
        lat_min_ = lat_min
        lat_max_ = lat_max

    if lon_sub:
        assert lon_sub[0] <= lon_sub[1], 'wrong latitude order'
        lon_min_ = lon_sub[0]
        lon_max_ = lon_sub[1]
    else:
        lon_min_ = lon_min
        lon_max_ = lon_max

    assert (lat_min_ >= lat_min) and (lat_max_ <= lat_max) and (lon_min_ >= lon_min) and (lon_max_ <= lon_max)

    # load full am series dataset

    outfile = DATADIR + '/am_series/domain_wise/' + ds + '/' + source + '/' + experiment + '/' +  member + '/' + res_ + '/lat({0},{1})'.format(lat_min, lat_max) + '_lon({0},{1})'.format(lon_min, lon_max) + '/amax'

    with open(outfile, 'rb') as pics:
        ams_all = dill.load(pics)

    coords = list(ams_all.keys())

    # sub-sample
    coords_sub = [coord for coord in coords if (lat_min_ <= coord[0] <= lat_max_) and (lon_min_ <= coord[1] <= lon_max_)]

    ams_sub = {}

    for coord in coords_sub:
        ams_sub[coord] = ams_all[coord]

    return ams_sub


def load_am_series_domain_date(ds='CMIP6', source='IPSL-CM6A-LR', experiment='historical', member='r1i1p1f1', lat_res=1.5, lon_res=2.3): #, var='pr', yr='1980', lat_sub=(lat_min_lr, lat_max_lr), lon_sub=(lon_min_lr, lon_max_lr)):
    """Load AM series dates for a given domain"""
    res_ = str(lat_res) + "x" + str(lon_res)

    assert lat_sub[0] <= lat_sub[1], 'wrong latitude order'
    lat_min_ = lat_sub[0]
    lat_max_ = lat_sub[1]

    assert lon_sub[0] <= lon_sub[1], 'wrong latitude order'
    lon_min_ = lon_sub[0]
    lon_max_ = lon_sub[1]

    # load full am series dataset
    outfile = DATADIR + '/am_series/domain_wise/' + ds + '/' + source + '/' + experiment + '/' + member + '/' + res_ + '/lat({0},{1})'.format(lat_min_lr, lat_max_lr) + '_lon({0},{1})'.format(lon_min_lr, lon_max_lr) + '/dmax'

    with open(outfile, 'rb') as pics:
        ams_date_all = dill.load(pics)

    # sub-sample: A FINIR

    coords = get_coords(ds, src, exp, member, lat_res, lon_res, 'pr', '1980')   # , lats=lat_sub, lons=lon_sub
    lats = coords[0]
    lons = coords[1]


    lats = np.arange(lat_min_, lat_max_, lat_res)
    lons = np.arange(lon_min_, lon_max_, lon_res)

    output = {}

    for lat_ in lats:
        lat_key = (float(lat_), float(lat_+lat_res))

        for lon_ in lons:
            lon_key = (float(lon_), float(lon_+lon_res))

            coord = (lat_key, lon_key)  #'({0},{1})'.format(lat_, lon_)
            output[coord] = ams_date_all[coord]

    return output


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

    #th_wd = TH_WD[src]

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

                    '''rrs = [out_rr[:,ilat,ilon] for out_rr in out_maxs]
                    rrs_ = np.concatenate(rrs, axis=0)
                    rr_nd = np.convolve(rrs_, kernel, mode='same')
                    rr_nd_max = np.nanmax(rr_nd, axis=0)'''

                    #rr_ = [pd.Series(out_rr[:,ilat,ilon], index=pd.date_range(start=years[iout_rr]+'-01-01', end=years[iout_rr]+'-12-31', freq='D')) for iout_rr, out_rr in enumerate(out_rrs)]
                    rr_ = [pd.Series(out_rr[:,ilat,ilon], index=dates[dates.year == int(years[iout_rr])]) for iout_rr, out_rr in enumerate(out_rrs)]
                    rrs_ = pd.concat(rr_, axis=0)
                    rrs_nd = rrs_.rolling('{0}D'.format(nd)).mean()
                    rrs_nd = rrs_nd.to_frame()
                    rrs_nd_ = rrs_nd.set_index([rrs_nd.index.year, rrs_nd.index.day])
                    rrs_nd_max = rrs_nd_.max(level=0)

                    df_am = pd.Series(data=rrs_nd_max.values.flatten(), index=years)
                    df_am = df_am.dropna()

                    """if len(df_am_.index) >= len(years) - int(0.25*len(years)):  # ALLOW 25% of missing values
                        out_ams[coord] = df_am
                    else:
                        out_ams[coord] = np.nan
                        sys.exit()"""

                    out_ams[coord] = df_am

                    #sys.exit()


        # GET DATE OF MAX

        #out_ams_date = {}
        #times = get_times(ds, src, exp, mem, lat_res, lon_res, var, y)

        #df = pd.Series(rains_, index=times)

        #df_ = df.rolling('1d').sum()
        #df_ = df_  / d_                        # cumul (mm) -> intensity (mm/h)

        #amax = df.max(axis=0)
        #amax_date = df.idxmax()
        #amax.append(df_.max(axis=0))          # get the annual maximum intensity at time step d
        #amax_date.append(df_.idxmax())        # get the date of annual maximum intensity at time step d

        #amax = np.array(amax, dtype='float64')

        """df_am.loc[y] = amax
        df_am_date.loc[y] = amax_date

        df_am = df_am.astype(dtype='float')

        out_ams[coord] = df_am
        out_ams_date[coord] = df_am_date"""


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


