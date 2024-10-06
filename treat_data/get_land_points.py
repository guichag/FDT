"""Select CMIP6 grid points on land"""

import os
import argparse
import pickle
import dill
import numpy as np

#from spatialobj import load_oceans, Point
from global_land_mask import globe

from dataa.d_config import lats_min, lats_max, lons_min, lons_max
from cmip6.treat_data.t_config import DATADIR
from cmip6.read_data.read_data import get_coords


### CST ###


### FUNC ###

def load_land_points_cmip(ds='CMIP6', source='IPSL-CM6A-LR', lat_res=1.27, lon_res=2.5, lat_sub=(-90., 90.), lon_sub=(-180., 180.), var='pr'):
    """return mask of grid points on ocean (True = ocean point)"""
    res_ = str(lat_res) + "x" + str(lon_res)

    lat_min = -90.  # lats_min[ds]
    lat_max = 90.   # lats_max[ds]
    lon_min = -180. # lons_min[ds]
    lon_max = 180.  # lons_max[ds]

    assert lat_sub[0] <= lat_sub[1], 'wrong latitude order'
    lat_min_ = lat_sub[0]
    lat_max_ = lat_sub[1]

    assert lon_sub[0] <= lon_sub[1], 'wrong latitude order'
    lon_min_ = lon_sub[0]
    lon_max_ = lon_sub[1]

    #assert (lat_min_ >= lat_min) and (lat_max_ <= lat_max) and (lon_min_ >= lon_min) and (lon_max_ <= lon_max)

    path = DATADIR + '/land_points/' + ds + '/' + source + '/' + res_ + '/' + var + '/ocean_mask_lat({0},{1})_lon({2},{3})'.format(lat_min_, lat_max_, lon_min_, lon_max_)

    with open(path, 'rb') as pics:
        output = dill.load(pics)

    return output


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument("--dataset", help="dataset", type=str, default='CMIP6')
    parser.add_argument("--source", help="source (GCM)", type=str, default='IPSL-CM6A-LR')
    parser.add_argument("--experiment", help="experiment (historical)", type=str, default='historical')
    parser.add_argument("--member", help="member", type=str, default='r1i1p1f1')
    parser.add_argument("--variable", help="variable to consider", type=str, default='pr')
    parser.add_argument("--lat_res", help="horizontal resolution (latitude)", type=float, default=1.27)
    parser.add_argument("--lon_res", help="horizontal resolution (longitude)", type=float, default=2.5)
    parser.add_argument("--lat_sub", nargs='+', help="min max latitude of domain", type=float, default=[-90., 90.])
    parser.add_argument("--lon_sub", nargs='+', help="min max longitude of domain", type=float, default=[-180., 180.])
    #parser.add_argument("--year", help="year", type=int, default=1850)

    opts = parser.parse_args()

    ds = opts.dataset
    src = opts.source
    exp = opts.experiment
    mem = opts.member
    var = opts.variable
    lat_res = opts.lat_res
    lon_res = opts.lon_res
    lat_sub = tuple(opts.lat_sub)
    assert lat_sub[0] <= lat_sub[1], 'wrong latitude order'
    lon_sub = tuple(opts.lon_sub)
    assert lon_sub[0] <= lon_sub[1], 'wrong longitude order'
    #year = opts.year


    lat_min = lats_min[ds]
    lat_max = lats_max[ds]
    lon_min = lons_min[ds]
    lon_max = lons_max[ds]

    lat_min_ = lat_sub[0]
    lat_max_ = lat_sub[1]
    lon_min_ = lon_sub[0]
    lon_max_ = lon_sub[1]

    #assert (lat_min_ >= lat_min) and (lat_max_ <= lat_max) and (lon_min_ >= lon_min) and (lon_max_ <= lon_max)

    out_lats = []

    #if res != None:

    coords = get_coords(ds=ds, source=src, experiment=exp, member=mem, lat_res=lat_res, lon_res=lon_res, var=var, yr='1950', lats=lat_sub, lons=lon_sub)

    lats = coords[0]   # Check center of grid cell
    lons = coords[1]


    print(lats)
    print(lons)

    print('Make Ocean mask for lat({0},{1}), lon({2},{3})'.format(lat_min_, lat_max_, lon_min_, lon_max_))


    for i in lats:
        print("\nlat: {0}".format(i))
        out_lons = []

        for j in lons:
            print(j, end=':', flush=True)            

            #p = Point(x=j, y=i)
            #test = o.search_point(p)
            test = globe.is_ocean(i, j)  # Ocean mask: True (masked) if point is ocean

            """if test.any() == False:   # False -> point not in the ocean
                cond = False
            else:
                cond = True"""

            out_lons.append(test)  # cond

        out_lats.append(out_lons)


    """elif res == None:
        coords = get_coords(ds, var, yr)
        lats = coords[0]
        lons = coords[1]

        for i in lats:
            print("\nlat: {0}".format(i))
            out_lons = []
            for j in lons:
                print(j, end=':', flush=True)            
                p = Point(x=j, y=i)
                test = o.search_point(p)

                if test.any() == False:   # False -> point not in the ocean
                    cond = False
                else:
                    cond = True

                out_lons.append(cond)

            out_lats.append(out_lons)

    else:
        print("Don't know what to do with {0}".format(res))"""


    mask = np.asarray(out_lats)


    # Save

    res_ = str(lat_res) + "x" + str(lon_res)

    if not os.path.isdir(DATADIR + '/land_points'):
        os.mkdir(DATADIR + '/land_points')
    outdir = DATADIR + '/land_points'

    if not os.path.isdir(outdir + '/' + ds):
        os.mkdir(outdir + '/' + ds)
    outdir = outdir + '/' + ds

    if not os.path.isdir(outdir + '/' + src):
        os.mkdir(outdir + '/' + src)
    outdir = outdir + '/' + src

    if not os.path.isdir(outdir + '/' + res_):
        os.mkdir(outdir + '/' + res_)
    outdir = outdir + '/' + res_

    if not os.path.isdir(outdir + '/' + var):
        os.mkdir(outdir + '/' + var)
    outdir = outdir + '/' + var

    outpath = outdir + '/ocean_mask_lat({0},{1})_lon({2},{3})'.format(lat_min_, lat_max_, lon_min_, lon_max_)

    with open(outpath, 'wb') as pics:
        pickle.dump(obj=mask, file=pics)


    print('\nDone')


