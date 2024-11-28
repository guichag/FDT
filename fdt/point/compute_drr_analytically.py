"""Compute DRR with analytical formula"""

import sys
import os
import argparse
import pickle
import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataa.d_config import lat_res, lon_res
from utils.make_nsgev import get_vdistr_norm
from cmip6.ns_gev.nsgev_config_cmip import OUTDIR
from cmip6.ns_gev.future.all_mems.fit_nsgev_freexi_future_all_mems_rversion_extRemes import load_nsgev_params_freexi_cmip_future_all_mems
from cmip6.ns_gev.future.all_mems.uncertainty.b_fit_boots_nsgev_all_mems import load_nsgev_unc_params_all_mems
from cmip6.fdt.fdt_config import DATADIR


### CST ###

yrefs = [2020]  # np.arange(2020, 2080+1, 1)
ts0 = np.arange(10., 100.+10, 10)  # np.linspace(10., 100., 50)


### FUNC ###

def get_y(xi, T):
    """Intermediary variable for analytical DRR calculation"""
    y = (1/xi) * ((-np.log(1-1/T))**-xi - 1)

    return y


def get_drr(loc_int, loc_sl, sca_int, sca_sl, xi, x, T0, T1):
    """Get DRR"""
    drr = (get_y(xi, T0) - get_y(xi, T1))*(x*sca_sl - loc_int*sca_sl + loc_sl*sca_int) / ((loc_sl+get_y(xi, T0)*sca_sl)*(loc_sl+get_y(xi, T1)*sca_sl))

    return drr


def load_analytical_drr(ds='CMIP6', source='CanESM5', experiment='ssp245', lat_res=2.79, lon_res=2.81, params=['loc', 'scale'], nmems=10, ymin=1950, ymax=2100, rrf=2., lat_sub=(-90., 90.), lon_sub=(-180., 180.), ndays=None):
    """Load DRR values computed analytically"""
    res = str(lat_res) + "x" + str(lon_res)
    params_ = "-".join(params)

    if ndays:
        nd_ = '_' + str(ndays) + 'd'
    else:
        nd_ = ''

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


    # load full drr dataset

    outdir = DATADIR + '/point/' + ds + '/' + source + '/' + experiment + '/' + res + '/' + params_ + '/analytical_drr'

    outfile = outdir + '/drr_' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmems) + '_rrf=' + str(rrf) + nd_ # + '_yref=' + str(yref) + '_' + ts_ + '-yr_nboots=' + str(nboots)

    with open(outfile, 'rb') as pics:
        drr_all = dill.load(pics)

    coords = list(drr_all.keys())

    # sub-sample
    coords_sub = [coord for coord in coords if (lat_min_ <= coord[0] <= lat_max_) and (lon_min_ <= coord[1] <= lon_max_)]

    drr_sub = {}

    for coord in coords_sub:
        drr_sub[coord] = drr_all[coord]

    return drr_sub


### MAIN ###


if __name__ == '__main__':

    #~ Get script parameters
    
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument("--dataset", help="dataset", type=str, default='CMIP6')
    parser.add_argument("--source", help="source (GCM)", type=str, default='CanESM5')
    parser.add_argument("--experiment", help="experiment (SSP)", type=str, default='ssp245')
    parser.add_argument("--nmembers", help="number of members to use", type=int, default=10)
    parser.add_argument("--params", nargs="+", help='params to be tested for non-stationary', type=str, default=['loc', 'scale'])
    parser.add_argument("--ymin", help='start of the period', type=int, default=1950)
    parser.add_argument("--ymax", help='end of the period', type=int, default=2100)
    #parser.add_argument("--yref", help='reference year', type=int, default=2020)
    parser.add_argument("--lat_pt", help='latitude of the point to plot', type=float, default=13.3)
    parser.add_argument("--lon_pt", help='longitude of the point to plot', type=float, default=2.1)
    parser.add_argument("--rrf", help='recurrence reduction factor', type=float, default=2.)
    parser.add_argument("--nboots", help='number of boot samples', type=int, default=100)
    parser.add_argument("--ndays", help="RXnD", type=int, default=None)

    opts = parser.parse_args()

    ds = opts.dataset
    src = opts.source
    ssp_exp = opts.experiment
    nmems = opts.nmembers
    params = opts.params
    ymin = opts.ymin
    ymax = opts.ymax
    #yref = opts.yref
    lat_pt = opts.lat_pt
    lon_pt = opts.lon_pt
    rrf = opts.rrf
    nboots = opts.nboots
    nd = opts.ndays

    params_ = "-".join(params)

    lat_res_ = lat_res[ds][ssp_exp][src]
    lon_res_ = lon_res[ds][ssp_exp][src]

    res_ = str(lat_res_) + "x" + str(lon_res_)

    years = np.arange(ymin, ymax+1, 1)

    if nd:
        nd_ = '_' + str(nd) + 'd'
    else:
        nd_ = ''

    """boots_dir = OUTDIR + '/data/boots/uncertainty/ns_all_mems/' + ds + '/' + src + '/' + ssp_exp + '/' + res_ + '/' + params_

    boots_files = [f for f in os.listdir(boots_dir) if ('pars' in f) and ('N={0}'.format(nmems) in f) and (str(ymax) in f)]

    if nmems < 10:
        n = 25
    else:
        n = 26

    boot_ranges = [f[n:] for f in boots_files]

    print(boot_ranges)"""


    #~ Outdir

    if not os.path.isdir(DATADIR + '/point'):
        os.mkdir(DATADIR + '/point')
    outdir = DATADIR + '/point'

    if not os.path.isdir(outdir + '/' + ds):
        os.mkdir(outdir + '/' + ds)
    outdir = outdir + '/' + ds

    if not os.path.isdir(outdir + '/' + src):
        os.mkdir(outdir + '/' + src)
    outdir = outdir + '/' + src

    if not os.path.isdir(outdir + '/' + ssp_exp):
        os.mkdir(outdir + '/' + ssp_exp)
    outdir = outdir + '/' + ssp_exp

    if not os.path.isdir(outdir + '/' + res_):
        os.mkdir(outdir + '/' + res_)
    outdir = outdir + '/' + res_

    if not os.path.isdir(outdir + '/' + params_):
        os.mkdir(outdir + '/' + params_)
    outdir = outdir + '/' + params_

    if not os.path.isdir(outdir + '/analytical_drr'):
        os.mkdir(outdir + '/analytical_drr')
    outdir = outdir + '/analytical_drr'


    #~ Get data

    print('-- Get data --')

    orig_pars = load_nsgev_params_freexi_cmip_future_all_mems(ds=ds, source=src, experiment=ssp_exp, nmembers=nmems, lat_res=lat_res_, lon_res=lon_res_, params=params, ymin=ymin, ymax=ymax, ndays=nd)

    """boots_pars_all = []
    for boot_range in boot_ranges:
        boots_pars = load_nsgev_unc_params_all_mems(ds=ds, source=src, experiment=ssp_exp, nmems=nmems, lat_res=lat_res_, lon_res=lon_res_, params=params, ymin=ymin, ymax=ymax, boot_range=boot_range)
        boots_pars_all.append(boots_pars)"""

    coords = list(orig_pars.keys())

    lats = np.unique([coord[0] for coord in coords])
    lons = np.unique([coord[1] for coord in coords])

    idx_lat_pt = (np.abs(lats - lat_pt)).argmin()
    idx_lon_pt = (np.abs(lons - lon_pt)).argmin()

    lat_pt_ = lats[idx_lat_pt]
    lon_pt_ = lons[idx_lon_pt]


    #~ Compute DRR

    print('-- Compute DRR --')

    out_drrs = {}
    #out_fails = {}

    for coord in coords:  #[(lat_pt_, lon_pt_)] , orig_pars_ in orig_pars.items():  [(-29.301359621762764, -67.5)]
        print('\n{0}'.format(coord))

        orig_pars_ = orig_pars[coord]

        if orig_pars_['succes'] == True:

            """boots_pars_ = [boot_pars[coord] for boot_pars in boots_pars_all]

            boots_pars__ = {}

            for boot_pars in boots_pars_:
                for boot, vals in boot_pars.items():
                    if vals['succes'] == True:
                        boots_pars__[boot] = vals

            nboots_ = len(boots_pars__.keys())

            #assert nboots_ >= nboots, 'Not enough boot samples'
            if nboots_ < nboots:
                #print(coord, end=' : ', flush=True)
                out_fails[coord] = nboots_"""

            orig_pars_.pop('succes')
            orig_pars_.pop('nllh')

            loc = orig_pars_['loc']
            scale = orig_pars_['scale']
            xi = -orig_pars_['c']

            nsd_orig = get_vdistr_norm('gev', orig_pars_, years)

            drrs_orig_yrs = {}

            for yref in yrefs:
                print(yref, end=' : ', flush=True)

                iyref = np.where(years == yref)[0][0]

                drrs_orig_ts = []
                #drrs_boots = []

                for t0 in ts0:
                    #print('\n  > {0}-yr'.format(t0))

                    rl0_orig = [d.return_level(t0) for d in nsd_orig][iyref]
                    drr_orig_norm = get_drr(loc[1], loc[0], scale[1], scale[0], xi, rl0_orig, t0, t0/rrf)
                    drr_orig = drr_orig_norm * (years[-1]-years[0])
                    drrs_orig_ts.append(drr_orig)

                    """drrs_boots_ = []

                    for boot in list(boots_pars__.keys())[:nboots]:   #boot_pars in boots_pars__.items():
                        print(boot, end=' : ', flush=True)

                        boot_pars = boots_pars__[boot]

                        loc_boot = boot_pars['loc']
                        scale_boot = boot_pars['scale']
                        xi_boot = -boot_pars['c']

                        nsd_boot = get_vdistr_norm('gev', boot_pars, years)

                        rl0_boot = [d.return_level(t0) for d in nsd_boot][iyref]
                        drr_boot_norm = get_drr(loc_boot[1], loc_boot[0], scale_boot[1], scale_boot[0], xi_boot, rl0_boot, t0, t0/rrf)
                        drr_boot = drr_boot_norm * (years[-1]-years[0])
                        drrs_boots_.append(drr_boot)

                    drrs_boots.append(drrs_boots_)"""

                """drrs_boots = np.asarray(drrs_boots)

                drrs_med = np.median(drrs_boots, axis=1)
                drrs_inf = np.percentile(drrs_boots, 5, axis=1)
                drrs_sup = np.percentile(drrs_boots, 95, axis=1)"""

                drrs_orig_ts = pd.Series(data=drrs_orig_ts, index=ts0)

                drrs_orig_yrs[yref] = drrs_orig_ts

            out_drrs[coord] = drrs_orig_yrs

        #else:
        #    drrs_orig_yrs = np.nan


    #~ Save

    print('\n-- Save --')

    outfile = outdir + '/drr_' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmems) + '_rrf=' + str(rrf) + nd_

    with open(outfile, 'wb') as pics:
        pickle.dump(obj=out_drrs, file=pics)


print('Done')

