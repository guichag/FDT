"""Plot map of point-wise T-yr to T'-yr duration of change (Duration of Recurrence Reduction)"""
"""With multi-member fit"""
"""Multi-model mean"""

import sys
import os
import argparse
import dill
import pickle
import scipy
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

from dataa.d_config import lat_res, lon_res
from cmip6.toe.point.compute_drr_analytically import load_analytical_drr
from cmip6.toe.toe_config import DATADIR, FIGDIR


### CST ###

cmap = 'plasma'
dbffile = './dataa/shapefiles/referenceRegions/referenceRegions.dbf'
cnydbffile = './dataa/shapefiles/World_Continents/World_Continents.dbf'   # ./dataa/world-boundaries/world-administrative-boundaries.dbf'
grid_step = 2.5  # horizontal resolution of new grid (grid_step x grid_step)
ts0 = np.arange(10., 100.+10, 10)


### FUNC ###

def add_matrix_NaNs(regridder):
    # Add NaN values outside of the grid, otherwise it puts 0 (see issue just up #15)
    X = regridder.weights
    M = scipy.sparse.csr_matrix(X)
    num_nonzeros = np.diff(M.indptr)
    M[num_nonzeros == 0, 0] = np.NaN
    regridder.weights = scipy.sparse.coo_matrix(M)
    return regridder


def load_regridded_analytical_drr(ds='CMIP6', source='CanESM5', ssp_exp='ssp245', grid_step=2.5, params=['loc', 'scale'], ymin=1950, ymax=2100, yref=2020, nmems=10, rrf=2., ndays=None):
    """Load regridded DRR data"""
    res_all = str(grid_step) + 'x' + str(grid_step)
    params_ = "-".join(params)

    if nd:
        nd_ = '_' + str(nd) + 'd'
    else:
        nd_ = ''

    outdir = DATADIR + '/maps/' + ds + '/drr_regridded/' + ssp_exp + '/' + res_all + '/' + params_ + '/DRR'
    outfile = outdir + '/global_drr_' + source + '_' + str(ymin) + '-' + str(ymax) + '_yref=' + str(yref) + '_N=' + str(nmems) + '_rrf=' + str(rrf) + nd_ + '.nc'

    out = xr.open_dataarray(outfile)

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
    parser.add_argument("--yref", help='reference year', type=int, default=2020)
    parser.add_argument("--rrf", help='Rec. red. factor', type=float, default=2.)
    parser.add_argument("--t0_2plot", help='return periods to plot', type=float, default=100.)
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
    yref = opts.yref
    rrf = opts.rrf
    t0_2plot = opts.t0_2plot
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
    iyref = np.where(years == yref)[0][0]

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

    xlbs = [ar6regs[ar6regs['Name'] == reg]['Acronym'].values[0] for reg in reg2plot]

    res_all = str(grid_step) + 'x' + str(grid_step)
    ds_out = xe.util.grid_global(grid_step, grid_step)

    classes = [[1, 20], [21, 40], [41, 60], [61, 80], [81, 150000000], [-10000000, 0]]
    pielbs = ["-".join([str(clas) for clas in classe]) for classe in classes[:-2]] + ['$>$ 80'] + ['$<$ 0 trend']

    #~ Outdir

    if not os.path.isdir(DATADIR + '/maps'):
        os.mkdir(DATADIR + '/maps')
    outdir = DATADIR + '/maps'

    if not os.path.isdir(outdir + '/' + ds):
        os.mkdir(outdir + '/' + ds)
    outdir = outdir + '/' + ds

    if not os.path.isdir(outdir + '/drr_regridded'):
        os.mkdir(outdir + '/drr_regridded')
    outdir = outdir + '/drr_regridded'

    if not os.path.isdir(outdir + '/' + ssp_exp):
        os.mkdir(outdir + '/' + ssp_exp)
    outdir = outdir + '/' + ssp_exp

    if not os.path.isdir(outdir + '/' + res_all):
        os.mkdir(outdir + '/' + res_all)
    outdir = outdir + '/' + res_all

    if not os.path.isdir(outdir + '/' + params_):
        os.mkdir(outdir + '/' + params_)
    outdir = outdir + '/' + params_
    
    if not os.path.isdir(outdir + '/DRR'):
        os.mkdir(outdir + '/DRR')
    outdir = outdir + '/DRR'


    #~ Figdir

    if not os.path.isdir(FIGDIR + '/maps'):
        os.mkdir(FIGDIR + '/maps')
    figdir = FIGDIR + '/maps'

    if not os.path.isdir(figdir + '/' + ds):
        os.mkdir(figdir + '/' + ds)
    figdir = figdir + '/' + ds

    if not os.path.isdir(figdir + '/drr_regridded'):
        os.mkdir(figdir + '/drr_regridded')
    figdir = figdir + '/drr_regridded'

    if not os.path.isdir(figdir + '/' + ssp_exp):
        os.mkdir(figdir + '/' + ssp_exp)
    figdir = figdir + '/' + ssp_exp

    if not os.path.isdir(figdir + '/' + res_all):
        os.mkdir(figdir + '/' + res_all)
    figdir = figdir + '/' + res_all

    if not os.path.isdir(figdir + '/amax'):
        os.mkdir(figdir + '/amax')
    figdir = figdir + '/amax'

    if not os.path.isdir(figdir + '/' + params_):
        os.mkdir(figdir + '/' + params_)
    figdir = figdir + '/' + params_

    if not os.path.isdir(figdir + '/DRR'):
        os.mkdir(figdir + '/DRR')
    figdir = figdir + '/DRR'

    if not os.path.isdir(figdir + '/' + domain):
        os.mkdir(figdir + '/' + domain)
    figdir = figdir + '/' + domain

    if not os.path.isdir(figdir + '/analytical'):
        os.mkdir(figdir + '/analytical')
    figdir = figdir + '/analytical'

    if not os.path.isdir(figdir + '/' + nd__):
        os.mkdir(figdir + '/' + nd__)
    figdir = figdir + '/' + nd__


    #~ Plot params

    cmap = plt.get_cmap(cmap)
    cmap.set_under(color='grey', alpha=0.975)
    cmap.set_over(color='grey', alpha=0.25)
    cmap_fail = plt.cm.get_cmap('Greys')

    ncols = [0., 10., 20., 30., 40., 50., 60., 70., 80.]  # np.arange(0, len(years[iyref:]), 10)   # np.linspace(-d_max, d_max, 41)
    norm = BoundaryNorm(ncols, ncolors=cmap.N) #, clip=True)

    xtcks = np.arange(-120., 120+60., 60.)  # lon_min_, lon_max_+lon_step, lon_step)
    ytcks = np.arange(-60., 60.+30., 30.)  # lat_min_, lat_max_+lat_step, lat_step)

    ts2plot = str(t0_2plot/rrf) + '.' + str(t0_2plot)


    #~ Get data

    print('-- Get data --')  # Not necessary if analytical DRR values already regridded

    fig, ax = plt.subplots(nrows=2, ncols=2)    # Plot individual models before regridding
    fig2, ax2 = plt.subplots(nrows=2, ncols=2)    # Plot individual models after regridding

    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, hspace=0.15)

    out_drrs_srcs_LR = []

    m = 0
    n = 0
    p = 0
    q = 0

    for src in srcs:
        print('\n>>> {0} <<<'.format(src))

        lat_res_ = lat_res[ds][ssp_exp][src]
        lon_res_ = lon_res[ds][ssp_exp][src]

        res_ = str(lat_res_) + "x" + str(lon_res_)

        data = load_analytical_drr(ds=ds, source=src, experiment=ssp_exp, nmems=nmems, lat_res=lat_res_, lon_res=lon_res_, params=params, ymin=ymin, ymax=ymax, rrf=rrf, ndays=nd)

        coords = list(data.keys())

        lats = np.unique([coord[0] for coord in coords])
        lons = np.unique([coord[1] for coord in coords])

        lons_ = lons - lon_res_ / 2
        lats_ = lats - lat_res_ / 2

        out_drrs_LR = []

        for t0 in ts0:
            print(t0, end=' : ', flush=True)

            out_drrs = []

            for ilat, lat in enumerate(lats):
                out_drrs_ = []

                for ilon, lon in enumerate(lons):
                    coord = (lat, lon)

                    if coord in coords:
                        data_ = data[coord][yref].loc[t0]
                        out_drrs_.append(data_)
                    else:
                        out_drrs_.append(np.nan) #[np.nan for i in range(len(ts0))])

                out_drrs.append(out_drrs_)

            out_drrs = np.asarray(out_drrs)

            drrs = xr.DataArray(out_drrs, dims=['lat', 'lon'], coords=dict(lon=(["lon"], lons), lat=(["lat"], lats)))

            if t0 == t0_2plot:
                if n == 2:
                    n = 0
                    m=m+1
                ax_ = ax[m][n]

                ax_.pcolor(lons, lats, drrs.values, norm=norm, cmap=cmap)

                ax_.set_title('{0} (N={1} $-$ {2})'.format(src, nmems, ssp_exp))
                ax_.set_xlim(-178., 178.)  # lon_sub[0], lon_sub[1]
                ax_.set_ylim(-60., 88.)  # lat_sub[0], lat_sub[1]
                ax_.set_xticks(xtcks)
                ax_.set_yticks(ytcks)
                ax_.xaxis.set_ticks_position('both')
                ax_.yaxis.set_ticks_position('both')

                # Add continent boundaries

                for i in dfcny.index:
                    pol_cont = gpd.GeoSeries(dfcny.iloc[i]['geometry'])
                    pol_cont.boundary.plot(ax=ax_, linewidth=0.45, color='k', zorder=100)

                n=n+1


            #~ Regrid data
            regridder = xe.Regridder(drrs, drrs, 'bilinear', periodic=False, reuse_weights=True)
            drrs_regrid = regridder(drrs)
            regridder_LR = xe.Regridder(drrs, ds_out, 'bilinear', periodic=False, reuse_weights=True)
            regridder_LR = add_matrix_NaNs(regridder_LR)
            drrs_LR = regridder_LR(drrs)

            out_drrs_LR.append(drrs_LR)

            drrs_LR = drrs_LR.where(drrs_LR.lat >= -60)    # FOR TEST !!!!

            lats_LR = drrs_LR.lat
            lons_LR = drrs_LR.lon

            lats_ = lats_LR - grid_step / 2
            lons_ = lons_LR - grid_step / 2

            if t0 == t0_2plot:
                if q == 2:
                    q = 0
                    p=p+1
                ax2_ = ax2[p][q]

                ax2_.pcolor(lons_, lats_, drrs_LR.values, norm=norm, cmap=cmap)

                ax2_.set_title('{0} (N={1} $-$ {2})'.format(src, nmems, ssp_exp))
                ax2_.set_xlim(-178., 178.)  # lon_sub[0], lon_sub[1]
                ax2_.set_ylim(-60., 88.)  # lat_sub[0], lat_sub[1]
                ax2_.set_xticks(xtcks)
                ax2_.set_yticks(ytcks)
                ax2_.xaxis.set_ticks_position('both')
                ax2_.yaxis.set_ticks_position('both')

                # Add continent boundaries

                for i in dfcny.index:
                    pol_cont = gpd.GeoSeries(dfcny.iloc[i]['geometry'])
                    pol_cont.boundary.plot(ax=ax2_, linewidth=0.45, color='k', zorder=100)

                q=q+1

        out_drrs_LR = xr.concat(out_drrs_LR, dim='ts0')
        out_drrs_LR = out_drrs_LR.assign_coords(ts0=ts0)
        out_drrs_srcs_LR.append(out_drrs_LR)


        #~ Save

        outfile = outdir + '/global_drr_' + src + '_' + str(ymin) + '-' + str(ymax) + '_yref=' + str(yref) + '_N=' + str(nmems) + '_rrf=' + str(rrf) + nd_ + '.nc'
        out_drrs_LR.to_netcdf(outfile)


    cbtcklbs = [int(n) for n in ncols]

    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[:], extend='both', orientation='horizontal', label='FDT [years]', shrink=0.65)
    cb.set_ticklabels(cbtcklbs)
    cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax2[:], extend='both', orientation='horizontal', label='FDT [years]', shrink=0.65)
    cb2.set_ticklabels(cbtcklbs)


    fig.set_size_inches(10., 6.5)
    fig2.set_size_inches(10., 6.5)
    fig.show()
    fig2.show()

    figfile_multi = figdir + '/' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmems) + '_yref=' + str(yref) + '_' + ts2plot + '_multi.pdf'
    figfile_multi_LR = figdir + '/' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmems) + '_yref=' + str(yref) + '_' + ts2plot + '_multi_LR.pdf'
    fig.savefig(figfile_multi)
    fig2.savefig(figfile_multi_LR)


    out_drrs = []

    for src in srcs:
       drrs = load_regridded_analytical_drr(ds, src, ssp_exp, grid_step, params, ymin, ymax, yref, nmems, rrf)
       out_drrs.append(drrs)

    out_drrs = xr.concat(out_drrs, dim='GCM')
    out_drrs_mean = out_drrs.mean(dim='GCM', skipna=True)   # TEST SKIPNA


    lats_LR = out_drrs_mean.lat
    lons_LR = out_drrs_mean.lon

    mask_land = rm.defined_regions.ar6.land.mask(lons_LR, lats_LR)  # AR6 regions mask
    mask_ocean = globe.is_ocean(mask_land.lat, mask_land.lon)   # Ocean mask: True (masked) if point is ocean

    mask_land_ = np.ma.array(mask_land, mask=mask_ocean)  # AR6 regions land points
    n_land = len(mask_land_.mask[mask_land_.mask == False])   # number of land grid points


    #~ Plot

    print('\n-- Plot map --')

    lons_ = lons_LR - grid_step / 2
    lats_ = lats_LR - grid_step / 2

    fig, ax = plt.subplots()  #subplot_kw={'projection': ccrs.PlateCarree()})

    plt.subplots_adjust(bottom=0.1, top=0.9)

    ax.pcolor(lons_, lats_, out_drrs_mean.sel(ts0=t0_2plot).values, norm=norm, cmap=cmap)

    ax.set_title('MMM (N={0} $-$ {1})'.format(nmems, ssp_exp))  # int(t0_2plot/rrf), int(t0_2plot), 
    ax.set_xlim(-178., 178.)  # lon_sub[0], lon_sub[1]
    ax.set_ylim(-60., 88.)  # lat_sub[0], lat_sub[1]
    ax.set_xticks(xtcks)
    ax.set_yticks(ytcks)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    # Add continent boundaries

    for i in dfcny.index:
        pol_cont = gpd.GeoSeries(dfcny.iloc[i]['geometry'])
        pol_cont.boundary.plot(ax=ax, linewidth=0.45, color='k', zorder=100)


    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, extend='both', orientation='horizontal', label='FDT [years]', shrink=0.75, pad=0.1)
    cb.set_ticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80])


    print('\n-- Bar plots --')

    reg2plot_ = ar6regs['Name']  # .drop([7, 15, 27, 39, 41], axis=0)['Name']   FOR TEST

    drrs_regs = {}
    drrs_regs_classes = {}

    fig_bar, ax_bar = plt.subplots(nrows=7, ncols=6)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0.25)

    xtlbs = [int(t) for t in ts0]
    ytcks_bar = np.arange(0, 100+20, 20)
    ytlbs_bar = ytcks_bar

    normcols = mpl.colors.Normalize(vmin=1, vmax=80)
    classes_center = [int((clas[1] - clas[0])/2 + clas[0]) for clas in classes[:-2]]
    rgbacols = [cmap(normcols(cent), bytes=True) for cent in classes_center]
    hexcols = ['#%02x%02x%02x' % (rgbacol[0], rgbacol[1], rgbacol[2]) for rgbacol in rgbacols]

    rgbalower = cmap_fail(99, bytes=True)
    hexcollower = '#%02x%02x%02x' % (rgbalower[0], rgbalower[1], rgbalower[2])
    rgbaupper = cmap_fail(50, bytes=True)
    hexcolupper = '#%02x%02x%02x' % (rgbaupper[0], rgbaupper[1], rgbaupper[2])
    hexcols = hexcols + [hexcolupper] + [hexcollower]

    i=0
    j=0

    out_prop_regs = {}
    prop_tot = []

    for ir, reg in enumerate(reg2plot_):
        print(reg, end=' : ', flush=True)

        ireg = ar6regs[ar6regs['Name'] == reg].index[0]
        mask_reg = np.ma.masked_not_equal(mask_land_, ireg)  # Land grid points in AR6 region
        n_land_reg = len(mask_reg.mask[mask_reg.mask == False])

        print(len(np.where(mask_land_ == ireg)[1]), n_land_reg)

        prop_n_land_reg = n_land_reg / n_land * 100

        prop_tot.append(n_land_reg)  # prop_n_land_reg

        acronym = ar6regs[ar6regs['Name'] == reg]['Acronym'][ireg]  # + ' ({0}%)'.format(round(prop_n_land_reg, 1))

        df_reg_srcs = []

        for isrc, src in enumerate(srcs):
            df_src = pd.DataFrame(index=ts0, columns=pielbs)

            for t0 in ts0:
                out_drrs_src = out_drrs.sel(ts0=t0).isel(GCM=isrc)

                masked_drrs = np.ma.array(out_drrs_src, mask=mask_reg.mask)
                drrs_reg = masked_drrs[~masked_drrs.mask].data.flatten()

                df = pd.DataFrame(data=drrs_reg, columns=['val']).dropna().round(0)

                if len(df) != 0:
                    prop_classes = [len(df.query('{0} <= val <= {1}'.format(classe[0], classe[1])).values) / len(df) * 100 for classe in classes]
                    df_src.loc[t0] = prop_classes

                    #print('{0} ({1}) : {2}'.format(reg, acronym, prop_classes))

                else:
                    print('{0}: NO {1} DATA'.format(reg, t0))

            df_reg_srcs.append(df_src)

        df_reg_srcs = pd.concat(df_reg_srcs)
        df_reg_mean = df_reg_srcs.groupby(df_reg_srcs.index).apply(lambda x: x.mean())


        if len(df_reg_mean.dropna()) != 0:
            drrs_regs_classes[reg] = df_reg_mean.loc[t0_2plot].values

            if j == len(ax_bar[1]):
                j=0
                i=i+1

            ax_ = ax_bar[i][j]

            df_reg_mean.plot.bar(ax=ax_, color=hexcols, stacked=True, width=1.05, legend=False)

            ax_.set_xticklabels([])
            ax_.set_xlim(-0.5, 9.5)
            ax_.set_yticks(ytcks_bar)
            ax_.set_yticklabels([])
            ax_.set_ylim(0., 100.)
            ax_.grid(True, axis='y', alpha=0.25)

            if j == 0:
                ax_.set_yticklabels(ytlbs_bar)
                ax_.set_ylabel('%')
            if i == ax_bar.shape[0]-1:
                ax_.set_xticklabels(xtlbs)
                ax_.set_xlabel('T$_0$ [years]')

            ax_.set_title(acronym)

            j=j+1

            out_prop_regs[reg] = df_reg_mean

        else:
            print('{0}: NO DATA AT ALL'.format(reg, t0))

    #fig_bar.delaxes(ax_bar[i][j])

    tab = pd.concat(out_prop_regs.values())
    tab_ = tab.loc[t0_2plot]
    tab_.index = out_prop_regs.keys()
    tab_ = tab_.astype(float).round(1)
    mean_prop = tab_.mean(axis=0)

    tabfile = outdir + '/global_drr_t0=' + str(t0_2plot) + '_rrf=' + str(rrf) + nd_

    tab_.to_latex(tabfile)


    # SUMMARY TABLE

    ts0_ = [int(t) for t in ts0]

    tab_sum = pd.DataFrame(index=ts0_, columns=pielbs)

    for t0 in ts0:
        sum_srcs = []
        for isrc, src in enumerate(srcs):
            out_drrs_ = out_drrs.sel(ts0=t0).isel(GCM=isrc)

            df = pd.DataFrame(data=out_drrs_.values.flatten(), columns=['val']).dropna().round(0)

            if len(df) != 0:
                prop_classes = [len(df.query('{0} <= val <= {1}'.format(classe[0], classe[1])).values) / len(df) * 100 for classe in classes]

            sum_srcs.append(prop_classes)

        sum_srcs_mean = np.mean(sum_srcs, axis=0)
        tab_sum.loc[t0] = sum_srcs_mean

    tab_sum = tab_sum.astype(float).round(1)

    tab_sum_file = outdir + '/global_drr_rrf=' + str(rrf) + '_summary' + nd_
    tab_sum.to_latex(tab_sum_file)


    print('\n-- Plot pie map --')

    fig_regs, ax_regs = plt.subplots()

    cols_regs = ['b', 'magenta', 'g', 'cyan', 'r', 'purple', 'k', 'olive', 'lime']

    for i in dfcny.index:
        pol_cont = gpd.GeoSeries(dfcny.iloc[i]['geometry'])
        pol_cont.boundary.plot(ax=ax_regs, linewidth=0.45, color='k', zorder=100)

    for reg, class_vals in drrs_regs_classes.items():
        ireg = ar6regs[ar6regs['Name'] == reg].index[0]
        acronym = ar6regs[ar6regs['Name'] == reg]['Acronym'][ireg]
        pol_reg = gpd.GeoSeries(ar6regs[ar6regs['Name'] == reg]['geometry'])
        pol_reg.boundary.plot(ax=ax_regs, linewidth=0.95, color='k', alpha=0.75)  # cols_regs[i] , label=acronym
        pol_reg.boundary.plot(ax=ax, linewidth=0.95, color='k', alpha=0.75)  # cols_regs[i] , label=acronym
        centroid = ar6regs.centroid[ireg]
        bounds = ar6regs.iloc[ireg]['geometry'].bounds

        ax_pie = inset_axes(ax_regs, width=0.3, height=0.3, bbox_to_anchor=(centroid.x-9.25, centroid.y-9.25), bbox_transform=ax_regs.transData, loc=3)
        ax_pie.pie(class_vals, colors=hexcols, labels=[' ', ' ',' ',' ',' ',' '])   # labels=pielbs wedgeprops={'alpha':0.75}
        ax_pie.set_axis_off()

        ax_regs.text(centroid.x-6.5, centroid.y-3, acronym, zorder=1000)  # xmin-4.

    ax_regs.set_title('IPCC AR6 regions')

    ax_regs.set_xlim(-178., 178.)
    ax_regs.set_ylim(-60., 88.)

    ax_regs.set_xticks(xtcks)
    ax_regs.set_yticks(ytcks)
    ax_regs.xaxis.set_ticks_position('both')
    ax_regs.yaxis.set_ticks_position('both')


    hs, ls = ax_pie.get_legend_handles_labels()

    fig_bar.legend(handles=hs, labels=pielbs, loc='lower center', bbox_to_anchor=(0.5, 0.025), ncol=6)


    print('-- Save --')

    fig.set_size_inches(10., 6.5)
    fig.show()

    figfile = figdir + '/' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmems) + '_yref=' + str(yref) + '_' + ts2plot + '-yr.pdf'
    fig.savefig(figfile)

    fig_bar.set_size_inches(14.5, 16.)
    fig_bar.show()

    figfile_bar = figdir + '/' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmems) + '_yref=' + str(yref) + '_rrf=' + str(rrf) + '_regs-bar.pdf'
    fig_bar.savefig(figfile_bar)

    fig_regs.set_size_inches(10., 6.5)
    fig_regs.show()

    figfile_regs = figdir + '/' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmems) + '_yref=' + str(yref) + '_' + ts2plot + '_regs-pie.pdf'
    fig_regs.savefig(figfile_regs)

  
    print('Done')

