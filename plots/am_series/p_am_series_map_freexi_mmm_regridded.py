"""Plot AM series"""
"""With multi-member"""
"""Multi-model"""

import sys
import os
import argparse
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

from utils.misc import natural_keys
from dataa.d_config import INDIR, lat_res, lon_res
from cmip6.treat_data.t_config import DATADIR, FIGDIR
from cmip6.treat_data.make_domainwise_am_series_v2 import load_am_series_domain_v2


### CST ###

cmap = 'viridis_r'
#lat_step = 30 # 20 # -> for WA domain  30 -> for global domain
#lon_step = 60 # 20 # -> for WA domain  60 -> for global domain
dbffile = './dataa/shapefiles/referenceRegions/referenceRegions.dbf'
cnydbffile = './dataa/shapefiles/World_Continents/World_Continents.dbf'   # ./dataa/world-boundaries/world-administrative-boundaries.dbf'
grid_step = 2.5  # horizontal resolution of new grid (grid_step x grid_step)

ymin_hist = 1993
ymax_hist = 2014
ymin_ssp = 2015
ymax_ssp = 2022


### FUNC ###

def add_matrix_NaNs(regridder):
    # Add NaN values outside of the grid, otherwise it puts 0 (see issue just up #15)
    X = regridder.weights
    M = scipy.sparse.csr_matrix(X)
    num_nonzeros = np.diff(M.indptr)
    M[num_nonzeros == 0, 0] = np.NaN
    regridder.weights = scipy.sparse.coo_matrix(M)
    return regridder


def load_regridded_amax_series(ds='CMIP6', source='CanESM5', ssp_exp='ssp245', grid_step=2.5, ymin=1993, ymax=2022, nmems=10, ndays=None):
    """Load regridded DRR data"""
    res_all = str(grid_step) + 'x' + str(grid_step)

    if nd:
        nd_ = '_' + str(nd) + 'd'
    else:
        nd_ = ''

    outdir = DATADIR + '/am_series/domain_wise' + ds + '/ams_regridded/' + ssp_exp + '/' + res_all
    outfile = outdir + '/global_amax_' + source + '_' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmems) + nd_ + '.nc'

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


    #~ Outdir

    if not os.path.isdir(DATADIR + '/am_series'):
        os.mkdir(DATADIR + '/am_series')
    outdir = DATADIR + '/am_series'

    if not os.path.isdir(DATADIR + '/domain_wise'):
        os.mkdir(DATADIR + '/domain_wise')
    outdir = DATADIR + '/domain_wise'

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

    if not os.path.isdir(figdir + '/' + ds):
        os.mkdir(figdir + '/' + ds)
    figdir = figdir + '/' + ds

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


    #~ Plot params

    cmap = plt.get_cmap(cmap)
    cmap.set_under(color='grey', alpha=0.975)
    cmap.set_over(color='grey', alpha=0.25)
    cmap_fail = plt.cm.get_cmap('Greys')

    cmin = 0.
    cmax = 175.

    ncols = np.linspace(0., 172., 11)
    norm = BoundaryNorm(ncols, ncolors=cmap.N) #, clip=True)

    xtcks = np.arange(-120., 120+60., 60.)  # lon_min_, lon_max_+lon_step, lon_step)
    ytcks = np.arange(-60., 60.+30., 30.)  # lat_min_, lat_max_+lat_step, lat_step)

    ts2plot = str(t0_2plot/rrf) + '.' + str(t0_2plot)


    #~ Get data

    print('-- Get data --')  # Not necessary if analytical DRR values already regridded

    fig, ax = plt.subplots(nrows=2, ncols=2)    # Plot individual models before regridding
    fig2, ax2 = plt.subplots(nrows=2, ncols=2)    # Plot individual models after regridding

    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, hspace=0.15)

    out_amax_srcs_LR = {}

    m = 0
    n = 0
    p = 0
    q = 0

    for src in srcs:
        lat_res_ = lat_res[ds][ssp_exp][src]
        lon_res_ = lon_res[ds][ssp_exp][src]

        res_ = str(lat_res_) + "x" + str(lon_res_)

        data_path = INDIR + '/' + ds + '/data/' + src + '/historical'
        mems = os.listdir(data_path)
        mems.sort(key=natural_keys)
        mems = mems[:nmems]

        if q == 2:
            q = 0
            p = p+1

        out_amax_LR = []

        ax_ = ax[p][q]
        ax2_ = ax2[p][q]

        for mem in mems:

            data_hist = load_am_series_domain_v2(ds=ds, source=src, experiment='historical', member=mem, lat_res=lat_res_, lon_res=lon_res_, lat_sub=lat_sub, lon_sub=lon_sub, ymin=ymin_hist, ymax=ymax_hist, ndays=nd)
            data_ssp = load_am_series_domain_v2(ds=ds, source=src, experiment=ssp_exp, member=mem, lat_res=lat_res_, lon_res=lon_res_, lat_sub=lat_sub, lon_sub=lon_sub, ymin=ymin_ssp, ymax=ymax_ssp, ndays=nd)
            data_all = xr.concat([data_hist, data_ssp], dim='time')

            """coords_all = list(orig_pars.keys())
            coords = list(data.keys())

            lats = np.unique([coord[0] for coord in coords_all])
            lons = np.unique([coord[1] for coord in coords_all])

            lons_ = lons - lon_res_ / 2
            lats_ = lats - lat_res_ / 2

            out_drrs_LR = []"""


            #~ Regrid data
            regridder = xe.Regridder(data_all, data_all, 'bilinear', periodic=False, reuse_weights=True)
            data_regrid = regridder(data_all)
            regridder_LR = xe.Regridder(data_all, ds_out, 'bilinear', periodic=False, reuse_weights=True)
            regridder_LR = add_matrix_NaNs(regridder_LR)
            data_LR = regridder_LR(data_all)

            #data_LR = data_LR.where(data_LR >= 5.)

            if 'member_id' in list(data_LR.coords):
                data_LR = data_LR.reset_coords('member_id', drop=True)

            out_amax_LR.append(data_LR)

            data_LR_ts = data_LR.mean(dim='x').mean(dim='y')
            data_LR_field = data_LR.mean(dim='time')

            #out_drrs_LR.append(drrs_LR)
            #drrs_LR = drrs_LR.where(drrs_LR.lat >= -60)    # FOR TEST !!!!

            lats_LR = data_LR.lat
            lons_LR = data_LR.lon

            lats_ = lats_LR - grid_step / 2
            lons_ = lons_LR - grid_step / 2

            #sys.exit()

            data_LR_ts.plot(ax=ax2_, lw=0.75, label=mem)

            ax2_.set_title('{0} (N={1})'.format(src, nmems))


        q=q+1

        out_amax_LR = xr.concat(out_amax_LR, dim='members')
        out_amax_LR = out_amax_LR.assign_coords(members=mems)

        out_amax_LR_mean = out_amax_LR.mean(dim='members').mean(dim='time')
        #out_amax_LR_mean.plot(ax=ax2_, lw=1.5, color='k')

        out_amax_LR_means = np.round(float(out_amax_LR_mean.mean().values), 1)

        ax_.pcolor(lons_, lats_, out_amax_LR_mean.values, norm=norm, cmap=cmap)  # out_amax_LR_mean

        ax_.text(0.025, 0.025, out_amax_LR_means, transform=ax_.transAxes)

        ax_.set_title('{0} (N={1})'.format(src, nmems))
        ax_.set_xlim(-178., 178.)  # lon_sub[0], lon_sub[1]
        ax_.set_ylim(-60., 88.)  # lat_sub[0], lat_sub[1]
        ax_.set_xticks(xtcks)
        ax_.set_yticks(ytcks)
        ax_.xaxis.set_ticks_position('both')
        ax_.yaxis.set_ticks_position('both')

        print('\n>>> {0} ({1}, {2}) <<<'.format(src, np.floor(out_amax_LR_mean.min()), np.ceil(out_amax_LR_mean.max())))


        # Add continent boundaries

        for i in dfcny.index:
            pol_cont = gpd.GeoSeries(dfcny.iloc[i]['geometry'])
            pol_cont.boundary.plot(ax=ax_, linewidth=0.45, color='k', zorder=100)

        #~ Save

        outfile = outdir + '/global_amax_' + src + '_' + str(ymin_hist) + '-' + str(ymax_ssp) + '_N=' + str(nmems) + nd_ + '.nc'
        out_amax_LR.to_netcdf(outfile)


    cbtcklbs = [int(n) for n in ncols]

    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[:], extend='both', orientation='horizontal', label='RX1D [mm d-1]', shrink=0.75)
    cb.set_ticklabels(cbtcklbs)
    #cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax2[:], extend='both', orientation='horizontal', label='FDT [years]', shrink=0.65)
    #cb2.set_ticklabels(cbtcklbs)'''


    fig.set_size_inches(10., 6.5)
    fig2.set_size_inches(10., 6.5)
    fig.show()
    fig2.show()

    figfile_multi = figdir + '/' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmems) + '_maps' + '.pdf'
    figfile_multi_LR = figdir + '/' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmems) + '_time_series' + '.pdf'
    fig.savefig(figfile_multi)
    fig2.savefig(figfile_multi_LR)



    sys.exit()


    #out_drrs_srcs_LR_ = xr.concat(out_drrs_srcs_LR, dim='GCMs')

    out_ams = []

    for src in srcs:
       ams = load_regridded_amax_series(ds, src, ssp_exp, grid_step, ymin, ymax, nmems, nd)
       out_ams.append(drrs)

    out_ams = xr.concat(out_ams, dim='GCM')
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


    # Make colorbar

    """fig_cb, ax_cb = plt.subplots(figsize=(5.5, 1.))

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.5, top=0.8)

    cb = fig_cb.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cb, extend='both', orientation='horizontal', label='DRR [years]')
    cb.set_ticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80])

    fig_cb.show()"""


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
        #masked_reg = np.ma.array(mask_reg, mask=mask_ocean) # Land grid point in AR6 region

        n_land_reg = len(mask_reg.mask[mask_reg.mask == False])

        print(len(np.where(mask_land_ == ireg)[1]), n_land_reg)

        prop_n_land_reg = n_land_reg / n_land * 100

        prop_tot.append(n_land_reg)  # prop_n_land_reg

        acronym = ar6regs[ar6regs['Name'] == reg]['Acronym'][ireg]  # + ' ({0}%)'.format(round(prop_n_land_reg, 1))

        df_reg_srcs = []

        #sys.exit()

        for isrc, src in enumerate(srcs):
            df_src = pd.DataFrame(index=ts0, columns=pielbs)

            for t0 in ts0:
                out_drrs_src = out_drrs.sel(ts0=t0).isel(GCM=isrc)

                #out_drrs_mean_ = out_drrs_mean.sel(ts0=t0)
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
            # Try continuous
            #ax_.pcolor(df_reg_mean.swapaxes(0, 1).values, norm=norm, cmap=cmap)  # contourf

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

    #sys.exit()


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
        #x_ax = 0.5*(1+centroid.x/180)  # centroid.x
        #y_ax = 0.5*(1+centroid.y/90)   # centroid.y
        #minx = pol_reg.bounds['maxx'][ireg]
        #miny = pol_reg.bounds['maxy'][ireg]
        #print('{0}: {1} {2} ({3} {4})'.format(reg, centroid.x, centroid.y, x_ax, y_ax)) #, rgbacol, hexcol)

        ax_pie = inset_axes(ax_regs, width=0.3, height=0.3, bbox_to_anchor=(centroid.x-9.25, centroid.y-9.25), bbox_transform=ax_regs.transData, loc=3)
        ax_pie.pie(class_vals, colors=hexcols, labels=[' ', ' ',' ',' ',' ',' '])   # labels=pielbs wedgeprops={'alpha':0.75}
        ax_pie.set_axis_off()

        ax_regs.text(centroid.x-6.5, centroid.y-3, acronym, zorder=1000)  # xmin-4.

        #sys.exit()

    #ax_regs.set_title('MMM (N={0} $-$ {1})'.format(nmems, ssp_exp))
    ax_regs.set_title('IPCC AR6 regions')

    ax_regs.set_xlim(-178., 178.)
    ax_regs.set_ylim(-60., 88.)

    ax_regs.set_xticks(xtcks)
    ax_regs.set_yticks(ytcks)
    ax_regs.xaxis.set_ticks_position('both')
    ax_regs.yaxis.set_ticks_position('both')


    hs, ls = ax_pie.get_legend_handles_labels()

    #ax_regs.legend(handles=hs, labels=pielbs, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=6)
    fig_bar.legend(handles=hs, labels=pielbs, loc='lower center', bbox_to_anchor=(0.5, 0.025), ncol=6)


    #sys.exit()

    print('-- Save --')

    fig.set_size_inches(10., 6.5)
    fig.show()

    figfile = figdir + '/' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmems) + '_yref=' + str(yref) + '_' + ts2plot + '-yr' + '.pdf'
    fig.savefig(figfile)


    fig_bar.set_size_inches(14.5, 16.)
    fig_bar.show()

    figfile_bar = figdir + '/' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmems) + '_yref=' + str(yref) + '_rrf=' + str(rrf) + '_regs-bar' + '.pdf'
    fig_bar.savefig(figfile_bar)


    fig_regs.set_size_inches(10., 6.5)
    fig_regs.show()

    figfile_regs = figdir + '/' + str(ymin) + '-' + str(ymax) + '_N=' + str(nmems) + '_yref=' + str(yref) + '_' + ts2plot + '_regs-pie' + '.pdf'
    fig_regs.savefig(figfile_regs)


    sys.exit()


    # Make legend

    #rect_nofit = Rectangle
    """nofit_patch = mpatches.Patch(color='k', label='no fit')
    neg_tr_patch = mpatches.Patch(color='grey', label='negative trend')"""

    hs, ls = ax_pie.get_legend_handles_labels()

    fig_leg = plt.figure(figsize=(5.5, 0.5))

    ax_leg = fig_leg.legend(handles=hs, labels=pielbs, ncol=5, loc='center')

    #fig_leg.show()


    print('Done')

