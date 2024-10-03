"""Fit NS-GEV model on CMIP6 AM series with relaxed shape parameter"""
"""With multi-member data"""

import os
import sys
import argparse
import pickle
import dill
import json
import numpy as np
import pandas as pd
import rpy2.robjects as ro

from rpy2.robjects import pandas2ri

from utils.misc import natural_keys
from dataa.d_config import lat_res, lon_res
from cmip6.cmip6_config import ymin_hist, ymax_hist, ymin_ssp, ymax_ssp
from cmip6.treat_data.t_config import OUTDIR
from cmip6.treat_data.make_domainwise_am_series import load_am_series_domain
from cmip6.ns_gev.nsgev_config_cmip import DATADIR


### CST ###

pandas2ri.activate()

r = ro.r
r['source']('fevd_func.R')
fevd_s_func_r = ro.globalenv['make_fevd_s']
fevd_ns_func_r = ro.globalenv['make_fevd_ns']
returnlevel_func_r_norm = ro.globalenv['get_return_levels_norm']
returnlevel_func_r_boot = ro.globalenv['get_return_levels_boot']
pars_func_r = ro.globalenv['get_pars_with_ci']

method_mle = ro.StrVector(['MLE'])
method_gmle = ro.StrVector(['GMLE'])
method_lmom = ro.StrVector(['Lmoments'])
method_unc = ro.StrVector(["boot"]) # ['normal'])

#to_json = ro.globalenv['to_json']
ci_func_r_norm = ro.globalenv['get_ci_norm']
ci_func_r_boot = ro.globalenv['get_ci_boot']


### FUNC ###

def load_sgev_params_freexi_cmip_future_all_mems(ds='CMIP6', source='IPSL-CM6A-LR', experiment='ssp245', nmembers=10, lat_res=1.27, lon_res=2.5, ymin=1950, ymax=2100):
    """load fitted S-GEV model parameters"""
    res_ = str(lat_res) + "x" + str(lon_res)

    """assert lat_sub[0] <= lat_sub[1], 'wrong latitude order'
    lat_min_ = lat_sub[0]
    lat_max_ = lat_sub[1]

    assert lon_sub[0] <= lon_sub[1], 'wrong latitude order'
    lon_min_ = lon_sub[0]
    lon_max_ = lon_sub[1]"""

    # load full s-gev dataset
    outfile = DATADIR + '/params/s_all_mems/' + ds + '/' + source + '/' + experiment + '/' + res_ + '/' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmembers)

    with open(outfile, 'rb') as pics:
        sgev_all = dill.load(pics)

    # sub-sample
    """lats = np.arange(lat_min_, lat_max_, lat_res)
    lons = np.arange(lon_min_, lon_max_, lon_res)

    output = {}

    for lat_ in lats:
        lat_key = (float(lat_), float(lat_+lat_res))

        for lon_ in lons:
            lon_key = (float(lon_), float(lon_+lon_res))

            k = (lat_key, lon_key)  #'({0},{1})'.format(lat_, lon_)
            output[k] = nsgev_all[k]"""

    return sgev_all


def load_nsgev_params_freexi_cmip_future_all_mems(ds='CMIP6', source='IPSL-CM6A-LR', experiment='ssp245', nmembers=10, lat_res=1.27, lon_res=2.5, params=['loc', 'scale'], ymin=1950, ymax=2100, ndays=None):
    """load fitted NS-GEV model parameters"""
    params_ = "-".join(params)
    res_ = str(lat_res) + "x" + str(lon_res)

    if ndays:
        nd_ = '_' + str(ndays) + 'd'
    else:
        nd_ = ''

    """assert lat_sub[0] <= lat_sub[1], 'wrong latitude order'
    lat_min_ = lat_sub[0]
    lat_max_ = lat_sub[1]

    assert lon_sub[0] <= lon_sub[1], 'wrong latitude order'
    lon_min_ = lon_sub[0]
    lon_max_ = lon_sub[1]"""

    # load full s-gev dataset
    outfile = DATADIR + '/params/ns_all_mems/' + ds + '/' + source + '/' + experiment + '/' + res_ + '/' + params_ + '/' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmembers) + nd_

    with open(outfile, 'rb') as pics:
        nsgev_all = dill.load(pics)

    # sub-sample
    """lats = np.arange(lat_min_, lat_max_, lat_res)
    lons = np.arange(lon_min_, lon_max_, lon_res)

    output = {}

    for lat_ in lats:
        lat_key = (float(lat_), float(lat_+lat_res))

        for lon_ in lons:
            lon_key = (float(lon_), float(lon_+lon_res))

            k = (lat_key, lon_key)  #'({0},{1})'.format(lat_, lon_)
            output[k] = nsgev_all[k]"""

    return nsgev_all


### MAIN ###

if __name__ == '__main__':

    #~ Get script parameters
    
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument("--dataset", help="dataset", type=str, default='CMIP6')
    parser.add_argument("--source", help="source (GCM)", type=str, default='IPSL-CM6A-LR')
    parser.add_argument("--experiment", help="experiment (SSP)", type=str, default='ssp245')
    parser.add_argument("--nmembers", help="number of members to use", type=int, default=10)
    parser.add_argument("--params", nargs="+", help='params to be tested for non-stationary', type=str, default=['loc', 'scale'])
    parser.add_argument("--model", help='AM distribution model', type=str, default="gev")
    parser.add_argument("--dist_method", help='distribution fit method', type=str, default="mle")
    parser.add_argument("--lat_sub", nargs='+', help="min max latitude of sub-domain", type=float, default=[-90., 90.])
    parser.add_argument("--lon_sub", nargs='+', help="min max longitude of sub-domain", type=float, default=[-180., 180.])
    parser.add_argument("--ymin", help='start of the period', type=int, default=1950)
    parser.add_argument("--ymax", help='end of the period', type=int, default=2100)
    parser.add_argument("--return_periods", nargs="+", help='return periods to compare', type=float, default=[2., 5., 10., 20., 50., 100.])

    opts = parser.parse_args()

    ds = opts.dataset
    src = opts.source
    ssp_exp = opts.experiment
    nmems = opts.nmembers
    params = opts.params
    model = opts.model
    dist_method = opts.dist_method
    lat_sub = opts.lat_sub
    lon_sub = opts.lon_sub
    ymin = opts.ymin
    ymax = opts.ymax
    ts = opts.return_periods
    
    ts_ = ro.FloatVector(ts)

    years = np.arange(ymin, ymax+1, 1)
    ynorm0 = (years - years.min()) / (years.max() - years.min())
    
    data_path_hist = OUTDIR + '/data/am_series/domain_wise/' + ds + '/' + src + '/historical'
    mems_hist = os.listdir(data_path_hist)

    data_path_ssp = OUTDIR + '/data/am_series/domain_wise/' + ds + '/' + src + '/' + ssp_exp
    mems_ssp = os.listdir(data_path_ssp)

    mems = [mem for mem in mems_hist if mem in mems_ssp]

    mems.sort(key=natural_keys)
    mems_ = mems[:nmems]

    params_ = "-".join(params)

    lat_res_hist = lat_res[ds]['historical'][src]
    lon_res_hist = lon_res[ds]['historical'][src]
    lat_res_ssp = lat_res[ds][ssp_exp][src]
    lon_res_ssp = lon_res[ds][ssp_exp][src]

    res_ = str(lat_res_hist) + "x" + str(lon_res_hist)

    print('Fit NS-GEV model {0} on {1}-{2} AMS'.format(params_, ymin, ymax))
    print('GCM: {0}'.format(src))
    print('Members: {0}'.format(mems_))
    print('Resolution: {0}'.format(res_))
    print('SSP scenario (y > {0}): {1}'.format(ymax_hist, ssp_exp))


    #~ Outdir

    if not os.path.isdir(DATADIR + '/params'):
        os.mkdir(DATADIR + '/params')
    datadir_params = DATADIR + '/params'
    
    if not os.path.isdir(datadir_params + '/s_all_mems'):
        os.mkdir(datadir_params + '/s_all_mems')
    datadir_params_s = datadir_params + '/s_all_mems'
    
    if not os.path.isdir(datadir_params_s + '/' + ds):
        os.mkdir(datadir_params_s + '/' + ds)
    datadir_params_s = datadir_params_s + '/' + ds
    
    if not os.path.isdir(datadir_params_s + '/' + src):
        os.mkdir(datadir_params_s + '/' + src)
    datadir_params_s = datadir_params_s + '/' + src

    if not os.path.isdir(datadir_params_s + '/' + ssp_exp):
        os.mkdir(datadir_params_s + '/' + ssp_exp)
    datadir_params_s = datadir_params_s + '/' + ssp_exp
    
    if not os.path.isdir(datadir_params_s + '/' + res_):
        os.mkdir(datadir_params_s + '/' + res_)
    datadir_params_s = datadir_params_s + '/' + res_


    if not os.path.isdir(datadir_params + '/ns_all_mems'):
        os.mkdir(datadir_params + '/ns_all_mems')
    datadir_params_ns = datadir_params + '/ns_all_mems'

    if not os.path.isdir(datadir_params_ns + '/' + ds):
        os.mkdir(datadir_params_ns + '/' + ds)
    datadir_params_ns = datadir_params_ns + '/' + ds

    if not os.path.isdir(datadir_params_ns + '/' + src):
        os.mkdir(datadir_params_ns + '/' + src)
    datadir_params_ns = datadir_params_ns + '/' + src

    if not os.path.isdir(datadir_params_ns + '/' + ssp_exp):
        os.mkdir(datadir_params_ns + '/' + ssp_exp)
    datadir_params_ns = datadir_params_ns + '/' + ssp_exp

    if not os.path.isdir(datadir_params_ns + '/' + res_):
        os.mkdir(datadir_params_ns + '/' + res_)
    datadir_params_ns = datadir_params_ns + '/' + res_

    if not os.path.isdir(datadir_params_ns + '/' + params_):
        os.mkdir(datadir_params_ns + '/' + params_)
    datadir_params_ns = datadir_params_ns + '/' + params_


    #~Get data

    print('-- Get data --')

    out_mems_ams = {}

    for mem in mems_:
        print('\n{0}'.format(mem))

        ams_hist = load_am_series_domain(ds=ds, source=src, experiment='historical', member=mem, lat_res=lat_res_hist, lon_res=lon_res_hist, lat_sub=lat_sub, lon_sub=lon_sub)
        ams_ssp = load_am_series_domain(ds=ds, source=src, experiment=ssp_exp, member=mem, lat_res=lat_res_ssp, lon_res=lon_res_ssp, lat_sub=lat_sub, lon_sub=lon_sub)

        coords_hist = list(ams_hist.keys())
        coords_ssp = list(ams_ssp.keys())

        if coords_hist >= coords_ssp:
            coords = coords_ssp
            coords = [coord for coord in coords if coord in coords_hist]
        elif coords_hist < coords_ssp:
            coords = coords_hist
            coords = [coord for coord in coords if coord in coords_ssp]

        out_ams = {}

        for coord in coords:
            #print(coord, end=' : ', flush=True)

            am_hist = ams_hist[coord]
            am_hist.index = [int(i) for i in am_hist.index]
            am_ssp = ams_ssp[coord]
            am_ssp.index = [int(i) for i in am_ssp.index]

            am_hist_ = am_hist.loc[ymin:ymax_hist]
            am_ssp_ = am_ssp.loc[ymin_ssp:ymax]

            am_ = pd.concat([am_hist_, am_ssp_], axis=0)

            out_ams[coord] = am_

        out_mems_ams[mem] = out_ams


    print('-- Get valid coordinates --')

    coords_all = [list(out_mem_ams.keys()) for out_mem_ams in out_mems_ams.values()]
    coords_all_ = [coord for coords_ in coords_all for coord in coords_]
    coords_all_ = pd.Series(coords_all_)
    coords_all_ = coords_all_.drop_duplicates().values

    val_coords = []
    for coord in coords_all_:
        k = 0
        for coords in coords_all:
            if coord in coords:
                k = k + 1

        if k == len(mems_):
            print(coord, end=' : ', flush=True)
            val_coords.append(coord)


    #~ Fit NS-GEV model

    print('\n-- Fit NS-GEV model --')

    out_s_pars = {}
    out_ns_pars = {}
    out_ns_pars_ci = {}

    fails_s = []
    fails_ns = []

    method = method_gmle    # method to use for NS-GEV fit -> MUST be MLE to get Confidence Intervals with Delta Method !!!!!!!!!!!!!!

    for coord in val_coords:
        print("\n{0}".format(coord))

        am_mems = [out_mems_ams[mem][coord] for mem in mems_]

        am_all = pd.concat(am_mems)
        am_all = am_all.sort_index()


        # Fit S-GEV model  -> get first guess parameter values for NLLH minimization

        sgev = fevd_s_func_r(am_all.values, method)
        sresults = sgev.rx2['results']

        spars = sresults[0]
        nllh_s = sresults[1][0]
        sconv = sresults[3]
        shess = sresults[5]

        sloc = spars[0]
        sscale = spars[1]
        sxi = spars[2]

        print('S-GEV NLLH: {0}'.format(nllh_s))


        # Fit NS model

        if (-0.5 <= sxi <= 0.5) and (0 < nllh_s < 100000):  # treat failed S-GEV fit !!!!!!!!!!!!!

            out_optim_s = {'succes': True, 'nllh': nllh_s, 'loc': sloc, 'scale': sscale, 'c': -sxi}
            out_s_fevd_ = sgev

            print('\nFit NS-GEV model')

            ynorm = (am_all.index - am_all.index.min()) / (am_all.index.max() - am_all.index.min())
            
            df_am_all = pd.DataFrame(data={'time': ynorm, 'am': am_all.values})  # , index=am_all.index

            am = ro.FloatVector(am_all.values) 

            data = pandas2ri.py2rpy(df_am_all)
            
            if 'loc' in params:
                fmla_loc = ro.Formula('~time')
            else:
                fmla_loc = ro.Formula('~1')

            if 'scale' in params:
                fmla_scale = ro.Formula('~time')
            else:
                fmla_scale = ro.Formula('~1')

            ar = np.array([1 for i in range(len(ynorm0))])   # to make qcov matrix for RL calculation


            print('Minimizing nllh')

            nsgev = fevd_ns_func_r(am, data, fmla_loc, fmla_scale, method)


            #~ Treat optim outputs

            nsresults = nsgev.rx2['results']

            nspars = nsresults[0]
            nllh_ns = nsresults[1][0]
            nsconv = nsresults[3]
            nshess = nsresults[5]

            res = nsconv[0]

            if ('loc' in params) and ('scale' not in params):

                c_val = -nspars[3]

                if (res == 0) and (-0.5 <= c_val <= 0.5) and (0 < nllh_ns < nllh_s):

                    succes = True
                    nllh_val = nllh_ns
                    loc_vals = [nspars[1], nspars[0]]
                    scale_vals = [nspars[2]]
                    qcov = np.array([ar, ynorm0, ar, ar, ar]).transpose()

                else:
                    succes = False
                    nllh_val = np.nan
                    loc_vals = [np.nan]
                    scale_vals = [np.nan]
                    c_val = np.nan


            elif ('loc' not in params) and ('scale' in params):

                c_val = -nspars[3]

                if (res == 0) and (-0.5 <= c_val <= 0.5) and (0 < nllh_ns < nllh_s):

                    succes = True
                    nllh_val = nllh_ns
                    loc_vals = [nspars[0]]
                    scale_vals = [nspars[2], nspars[1]]
                    qcov = np.array([ar, ar, ar, ynorm0, ar]).transpose()

                else:
                    succes = False
                    nllh_val = np.nan
                    loc_vals = [np.nan]
                    scale_vals = [np.nan]
                    c_val = np.nan


            elif ('loc' in params) and ('scale' in params):

                c_val = -nspars[4]  # [1]

                if (res == 0) and (-0.5 <= c_val <= 0.5) and (0 < nllh_ns < nllh_s):

                    succes = True
                    nllh_val = nllh_ns
                    loc_vals = [nspars[1], nspars[0]]  # [slope, interc]
                    scale_vals = [nspars[3], nspars[2]]  # [slope, interc]
                    qcov = np.array([ar, ynorm0, ar, ynorm0, ar]).transpose()

                else:
                    succes = False
                    nllh_val = np.nan
                    loc_vals = [np.nan]
                    scale_vals = [np.nan]
                    c_val = np.nan  # [np.nan]
                    fails_ns.append(coord)

            """if succes == True:    # compute NS RLS
                out_rls_ns_ = {}

                for t in ts:
                    rls_ns = returnlevel_func_r_norm(nsgev, t, qcov)
                    rls_ci = ci_func_r_norm(nsgev, t, qcov)
                    #rls_ci = ci_func_r_boot(nsgev, t)
                    df_rls_ns = pd.DataFrame(rls_ns, index=years)
                    out_rls_ns_[t] = df_rls_ns


                #out_rls_ = {'rls': rls[:,1], 'rls_inf': rls[:,0], 'rls_sup': rls[:,2]}
                #out_ns_fevd_ = nsgev  # {'succes': True, 'fevd': nsgev}

            else:
                #out_rls_ns_ = np.nan"""


            out_optim_ns = {'succes': succes, 'nllh': nllh_val, 'loc': loc_vals, 'scale': scale_vals, 'c': c_val}  # c_vals

            print('output: {0}'.format(out_optim_ns))


        else:
            out_optim_s = {'succes': False, 'nllh': np.nan, 'loc': np.nan, 'scale': np.nan, 'c': np.nan}
            out_optim_ns = {'succes': False, 'nllh': np.nan, 'loc': np.nan, 'scale': np.nan, 'c': np.nan}
            fails_s.append(coord)

            print('S-GEV fit failed')


        out_s_pars[coord] = out_optim_s
        out_ns_pars[coord] = out_optim_ns


    #~ Save

    print('\n-- Save --')

    outfile_s_params = datadir_params_s + '/' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmems)
    outfile_ns_params = datadir_params_ns + '/' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmems)

    with open(outfile_s_params, 'wb') as pics:
        pickle.dump(obj=out_s_pars, file=pics)

    with open(outfile_ns_params, 'wb') as pics:
        pickle.dump(obj=out_ns_pars, file=pics)

    print('Done')

