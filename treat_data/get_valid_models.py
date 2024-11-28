"""Get valid models"""
"""With multi-member"""
"""Multi-model"""

import sys
import os
import argparse
import scipy.stats
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import xesmf as xe
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import regionmask as rm

from global_land_mask import globe
from shapely.geometry import Point
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utils.misc import natural_keys
from dataa.d_config import INDIR, lat_res, lon_res
from cmip6.treat_data.t_config import DATADIR, FIGDIR
from cmip6.treat_data.make_domainwise_am_series_v2 import load_am_series_domain_v2
from cmip6.plots.am_series.p_am_series_map_freexi_mmm_regridded import *


### CST ###


### FUNC ###

def load_valid_models(ds='CMIP6', ssp_exp='ssp245', grid_step=2.5, ndays=None):
    """Load valid models"""
    res_all = str(grid_step) + 'x' + str(grid_step)

    if ndays:
        nd_ = '_' + str(ndays) + 'd'
    else:
        nd_ = ''

    outdir = DATADIR + '/am_series/domain_wise/' + ds + '/ams_regridded/' + ssp_exp + '/' + res_all
    outfile_vals_models = outdir + '/valid_models' + nd_ + '.pic'

    out = pd.read_pickle(outfile_vals_models)

    return out


### MAIN ###

if __name__ == '__main__':

    #~ Get script parameters

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument("--dataset", help="dataset", type=str, default='CMIP6')
    parser.add_argument("--sources", nargs="+", help="sources (GCM)", type=str, default=['ACCESS-ESM1-5', 'CanESM5', 'IPSL-CM6A-LR', 'MPI-ESM1-2-LR'])
    parser.add_argument("--experiment", help="experiment", type=str, default='ssp245')
    parser.add_argument("--nmembers", help="number of members", type=int, default=10)
    parser.add_argument("--params", nargs="+", help='params to be tested for non-stationary', type=str, default=['loc', 'scale'])
    parser.add_argument("--lat_sub", nargs='+', help="min max latitude of sub-domain", type=float, default=[-90., 90.])
    parser.add_argument("--lon_sub", nargs='+', help="min max longitude of sub-domain", type=float, default=[-180., 180.])
    parser.add_argument("--ymin", help='start of the period', type=int, default=1950)
    parser.add_argument("--ymax", help='end of the period', type=int, default=2100)
    parser.add_argument("--ndays", help="RXnD", type=int, default=None)

    opts = parser.parse_args()

    ds = opts.dataset
    srcs = opts.sources
    ssp_exp = opts.experiment
    nmems = opts.nmembers
    params = opts.params
    lat_sub = tuple(opts.lat_sub)
    lon_sub = tuple(opts.lon_sub)
    ymin = opts.ymin
    ymax = opts.ymax
    nd = opts.ndays

    params_ = "-".join(params)

    lat_min_ = lat_sub[0]
    lat_max_ = lat_sub[1]
    lon_min_ = lon_sub[0]
    lon_max_ = lon_sub[1]

    domain = '({0},{1})'.format(lat_min_, lat_max_) + '_({0},{1})'.format(lon_min_, lon_max_)

    if lat_max_ - lat_min_ == 20:
        lat_step = 20.
    elif lat_max_ - lat_min_ == 180:
        lat_step = 30.

    if lon_max_ - lon_min_ == 40:
        lon_step = 20.
    elif lon_max_ - lon_min_ == 360:
        lon_step = 60.

    yrs_spec = str(ymin) + '-' + str(ymax)

    years = np.arange(ymin, ymax+1, 1)

    if nd:
        nd_ = '_' + str(nd) + 'd'
        nd__ = str(nd) + 'd'
    else:
        nd_ = ''
        nd__ = ''


    df = gpd.read_file(dbffile)
    dfcny = gpd.read_file(cnydbffile)

    ar6regs = rm.defined_regions.ar6.df[:43]
    reg2plot = ar6regs['Name'].values

    reg2plot_ = ar6regs['Name']
    reg2plot_.drop(index=[8], inplace=True)  # drop Caraibes

    xlbs = [ar6regs[ar6regs['Name'] == reg]['Acronym'].values[0] for reg in reg2plot]

    res_all = str(grid_step) + 'x' + str(grid_step)
    ds_out = xe.util.grid_global(grid_step, grid_step)


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

    if not os.path.isdir(outdir + '/ams_regridded'):
        os.mkdir(outdir + '/ams_regridded')
    outdir = outdir + '/ams_regridded'

    if not os.path.isdir(outdir + '/' + ssp_exp):
        os.mkdir(outdir + '/' + ssp_exp)
    outdir = outdir + '/' + ssp_exp

    if not os.path.isdir(outdir + '/' + res_all):
        os.mkdir(outdir + '/' + res_all)
    outdir = outdir + '/' + res_all


    #~ Figdir

    if not os.path.isdir(FIGDIR + '/' + ds):
        os.mkdir(FIGDIR + '/' + ds)
    figdir = FIGDIR + '/' + ds

    if not os.path.isdir(figdir + '/ams_regridded'):
        os.mkdir(figdir + '/ams_regridded')
    figdir = figdir + '/ams_regridded'

    if not os.path.isdir(figdir + '/' + ssp_exp):
        os.mkdir(figdir + '/' + ssp_exp)
    figdir = figdir + '/' + ssp_exp

    if not os.path.isdir(figdir + '/' + res_all):
        os.mkdir(figdir + '/' + res_all)
    figdir = figdir + '/' + res_all

    '''if not os.path.isdir(figdir + '/' + nd__):
        os.mkdir(figdir + '/' + nd__)
    figdir = figdir + '/' + nd__'''


    #~ Get data

    print('-- Get data --')  # Not necessary if analytical DRR values already regridded

    '''ams_era5 = load_am_series_domain_v2(ds='reanalyses', source='ERA5', member='r1', lat_res=0.25, lon_res=0.25, lat_sub=lat_sub, lon_sub=lon_sub, ymin=ymin_hist, ymax=ymax_ssp, ndays=nd)
    ams_era5 = ams_era5.rename({'latitude': 'lat', 'longitude': 'lon'})

    #~ Regrid data
    regridder = xe.Regridder(ams_era5, ams_era5, 'bilinear', periodic=False, reuse_weights=True)
    data_regrid = regridder(ams_era5)
    regridder_LR = xe.Regridder(ams_era5, ds_out, 'bilinear', periodic=False, reuse_weights=True)
    regridder_LR = add_matrix_NaNs(regridder_LR)
    ams_era5_LR = regridder_LR(ams_era5)

    outfile = outdir + '/global_amax_era5_' + str(ymin_hist) + '-' + str(ymax_ssp) + nd_ + '.nc'
    era5_LR.to_netcdf(outfile)'''

    ams_era5 = load_regridded_amax_series('reanalyses', 'ERA5', 'historical', grid_step, ymin_hist, ymax_ssp, 1, nd)


    out_ams_srcs = []

    for src in srcs:
       ams = load_regridded_amax_series(ds, src, ssp_exp, grid_step, ymin_hist, ymax_ssp, nmems, nd)
       out_ams_srcs.append(ams)

    out_ams_srcs = xr.concat(out_ams_srcs, dim='GCM')
    out_ams_srcs = out_ams_srcs.assign_coords(GCM=srcs)

    lats_LR = out_ams_srcs.lat
    lons_LR = out_ams_srcs.lon

    mask_land = rm.defined_regions.ar6.land.mask(lons_LR, lats_LR)  # AR6 regions mask
    mask_ocean = globe.is_ocean(mask_land.lat, mask_land.lon)   # Ocean mask: True (masked) if point is ocean

    mask_land_ = np.ma.array(mask_land, mask=mask_ocean)  # AR6 regions land points
    n_land = len(mask_land_.mask[mask_land_.mask == False])   # number of land grid points


    #~ Plot trend vs mean / std

    stats = ['mean', 'std', 'trend']

    fig, ax = plt.subplots(nrows=7, ncols=6)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.25, hspace=0.275)

    i=0
    j=0

    stats_regs_era5 = pd.DataFrame(columns=stats, index=reg2plot_.values)
    stats_regs = {}

    for ir, reg in enumerate(reg2plot_):
        print(reg, end=' : ', flush=True)

        ireg = ar6regs[ar6regs['Name'] == reg].index[0]
        acronym = ar6regs[ar6regs['Name'] == reg]['Acronym'][ireg]  # + ' ({0}%)'.format(round(prop_n_land_reg, 1))

        mask_reg = np.ma.masked_not_equal(mask_land_, ireg)  # Land grid points in AR6 region
        mask_reg_ = np.broadcast_arrays(mask_reg.mask, ams_era5)[0]

        n_land_reg = len(mask_reg.mask[mask_reg.mask == False])

        print(len(np.where(mask_land_ == ireg)[1]), n_land_reg)

        # treat ERA5 data
        ams_era5_reg = ams_era5.where(~mask_reg_)

        mean_era5 = ams_era5_reg.mean(dim='year').mean().values
        std_era5 = ams_era5_reg.std(dim='year').mean().values

        ols_era5_reg = ams_era5_reg.polyfit(dim='year', deg=1)
        slopes = ols_era5_reg.polyfit_coefficients.isel(degree=0)
        intercs = ols_era5_reg.polyfit_coefficients.isel(degree=1)
        trends_era5_reg = slopes / (slopes*1993+intercs) * 10 * 100
        trend_era5_reg = trends_era5_reg.mean().values

        stats_regs_era5.loc[reg]['mean'] = mean_era5
        stats_regs_era5.loc[reg]['std'] = std_era5
        stats_regs_era5.loc[reg]['trend'] = trend_era5_reg

        if j == len(ax[1]):
            j=0
            i=i+1

        ax_ = ax[i][j]

        ax_.scatter(mean_era5, trend_era5_reg, color='k', s=7.5, alpha=0.75)

        k=0

        stats_srcs = {}

        for isrc, src in enumerate(srcs):
            ams_src = out_ams_srcs.sel(GCM=src)

            mems = ams_src.members.values

            df_stats = pd.DataFrame(columns=stats, index=range(len(mems)))

            for imem, mem in enumerate(mems):
                ams_mem = ams_src.sel(members=mem)

                ams_mem_reg = ams_mem.where(~mask_reg_)

                ams_mem_mean = ams_mem_reg.mean(dim='time').mean().values
                ams_mem_std = ams_mem_reg.std(dim='time').std().values

                ols_mem_reg = ams_mem_reg.polyfit(dim='time', deg=1)
                slopes_mem = ols_mem_reg.polyfit_coefficients.isel(degree=0)
                intercs_mem = ols_mem_reg.polyfit_coefficients.isel(degree=1)
                trends_mem = slopes_mem / (slopes_mem*1993+intercs_mem) * 10 * 100
                trend_mem = trends_mem.mean().values

                df_stats.loc[imem]['mean'] = ams_mem_mean
                df_stats.loc[imem]['std'] = ams_mem_std
                df_stats.loc[imem]['trend'] = trend_mem

            stats_srcs[src] = df_stats

        stats_regs[reg] = stats_srcs


        arr = np.array(list(stats_srcs.values()))

        mean_min = arr.min(axis=1)[:,0]  # np.percentile(arr, axis=1)[:,0]
        mean_mean = arr.mean(axis=1)[:,0]
        mean_max = arr.max(axis=1)[:,0]
        trend_min = arr.min(axis=1)[:,2]
        trend_mean = arr.mean(axis=1)[:,2]
        trend_max = arr.max(axis=1)[:,2]
        
        pearsr = scipy.stats.pearsonr(mean_mean, trend_mean)
        corr = pearsr[0]
        pval = pearsr[1]
        if pval > 0.1:
            pval_ = ''
        elif 0.05 < pval <= 0.1:
            pval_ = '*'
        elif 0.01 < pval <= 0.05:
            pval_ = '**'
        elif pval <= 0.01:
            pval_ = '***'

        xerr = [mean_mean-mean_min, mean_max-mean_mean]
        yerr = [trend_mean-trend_min, trend_max-trend_mean]

        ax_.errorbar(x=mean_mean, y=trend_mean, xerr=xerr, yerr=yerr, fmt='none', ecolor=['b', 'orange', 'g', 'r']) # , markerfacecolor=['b', 'orange', 'g', 'r']
        ax_.text(0.025, 0.9, '%.2f %s'%(corr, pval_), transform=ax_.transAxes)
        ax_.axhline(0, 0, 1, lw=0.75, color='k', alpha=0.5)

        ax_.set_title(acronym)

        if j == 0:
            ax_.set_ylabel('Trend [% decade$^{-1}$]')
        if i == ax.shape[0]-1:
            ax_.set_xlabel('Mean [mm d$^{-1}$]')

        j=j+1


    hs, ls = ax_.get_legend_handles_labels()

    fig.legend(handles=hs, labels=ls, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=5)

    fig.set_size_inches(14.5, 16.)
    fig.show()

    figfile = figdir + '/' + str(ymin_hist) + '-' + str(ymax_ssp) + '_N=' + str(nmems) + '_mean_vs_trend' + nd_ + '.pdf'
    fig.savefig(figfile)
    figfile = figdir + '/' + str(ymin_hist) + '-' + str(ymax_ssp) + '_N=' + str(nmems) + '_mean_vs_trend' + nd_ + '.png'
    fig.savefig(figfile)



    reg2plot_.drop(index=[36], inplace=True)  # drop Ar Peninsula (ERA5 outlier)


    # Plot stats (regions / GCM)

    vals_models = pd.DataFrame(columns=srcs, index=reg2plot_.values)

    fig, ax = plt.subplots(nrows=1, ncols=4)

    plt.subplots_adjust(left=0.045, right=0.99, bottom=0.125, top=0.95, wspace=0.125)

    xtcks = np.arange(1, reg2plot_.shape[0]+1, 1)

    for isrc, src in enumerate(srcs):
        xtlbs = []

        for ir, reg in enumerate(reg2plot_):
            ireg = ar6regs[ar6regs['Name'] == reg].index[0]
            acronym = ar6regs[ar6regs['Name'] == reg]['Acronym'][ireg]  # + ' ({0}%)'.format(round(prop_n_land_reg, 1))
            xtlbs.append(acronym)

            stats_era5 = stats_regs_era5.loc[reg]
            stats = stats_regs[reg][src]

            min_th = stats['trend'].quantile(0.125)  #  stats['trend'].min()
            max_th = stats['trend'].quantile(0.875)  #  stats['trend'].max()

            if min_th <= stats_era5['trend'] <= max_th:
                vals_models.loc[reg][src] = 1
            else:
                vals_models.loc[reg][src] = 0

            '''p0 = ax[isrc].scatter(np.ones(10)+ir, stats['mean'].values, alpha=0.2)
            col = p0.to_rgba(0)
            ax[isrc].scatter(1.+ir, stats.mean(axis=0)['mean'], color=col, alpha=0.85)
            ax[isrc].scatter(1.+ir, stats_era5['mean'], color='k')

            p1 = ax[1][isrc].scatter(np.ones(10)+ir, stats['std'].values, alpha=0.2)
            col = p1.to_rgba(0)
            ax[1][isrc].scatter(1.+ir, stats.mean(axis=0)['std'], color=col, alpha=0.85)
            ax[1][isrc].scatter(1.+ir, stats_era5['std'], color='k')'''

            p2 = ax[isrc].scatter(np.ones(10)+ir, stats['trend'].values, s=6.5, alpha=0.2)
            col = p2.get_facecolors()[0]  # np.array([p2.to_rgba(0)])
            ax[isrc].scatter(1.+ir, stats.median(axis=0)['trend'], color=col, s=6.5, alpha=0.85)
            ax[isrc].scatter(1.+ir, stats_era5['trend'], color='k', s=6.5, alpha=0.75)

        ax[isrc].axhline(0, 0, 1, lw=0.75, color='k', alpha=0.5)

        ax[isrc].set_xlim(xtcks[0]-0.5, xtcks[-1]+0.5)
        ax[isrc].set_ylim(-15., 45.)
        ax[isrc].set_xticks(xtcks)
        ax[isrc].set_xticklabels(xtlbs, rotation=90.)
        ax[isrc].tick_params(axis='x', labelsize=8.25)
        ax[isrc].tick_params(axis='y', rotation=45)
        ax[isrc].set_title(src)

    #ax[0].set_ylabel('mean [mm d$^{-1}$]')
    #ax[1].set_ylabel('std [mm d$^{-1}$]')
    ax[0].set_ylabel('Trend [% decade$^{-1}$]')

    fig.set_size_inches(14.5, 4.5)
    fig.show()

    figfile = figdir + '/' + str(ymin_hist) + '-' + str(ymax_ssp) + '_N=' + str(nmems) + '_stats' + nd_ + '.pdf'
    fig.savefig(figfile)
    figfile = figdir + '/' + str(ymin_hist) + '-' + str(ymax_ssp) + '_N=' + str(nmems) + '_stats' + nd_ + '.png'
    fig.savefig(figfile)


    outfile_vals_models = outdir + '/valid_models' + nd_ + '.pic'
    vals_models.to_pickle(outfile_vals_models)


    vals_regs = vals_models.sum(axis=1)
    vals_regs = vals_regs[vals_regs !=0]

    vals_models_file = outdir + '/valid_models' + nd_
    vals_regs.to_latex(vals_models_file)


print('Done')

