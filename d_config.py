"""Data config file"""

import os

DIRNAME = os.path.dirname(__file__)

DATADIR_ERA5 = os.path.join(DIRNAME, "ERA5")
DATADIR_CMIP6 = os.path.join(DIRNAME, "CMIP6")

DATA_BADO = os.path.join(DIRNAME, "BADOPLU")
DATA_BADO_RAINFIELDS = '/home/quanting/BDD/data/badoplu/rainfields' #/day/west_africa'  #/0.25x0.25'  #/day/west_africa/JRC_1x1   # os.path.join(DATA_BADO, "rainfields") #

DATADIR_AC = os.path.join(DIRNAME, "AC")

DATADIR_SAT = os.path.join(DIRNAME, "SAT") # sat2nc -r 0.25 -s 1983-01-01_0:0:0 -e 1983-12-31_0:0:0 CHIRPS 025_1983.nc

DATADIR_discharge = os.path.join(DIRNAME, "discharge")


### CST ###

#~ periods

ac_periods = {'niger': 28, 'benin': 19}

era5_years = ['1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987',
         '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996',
         '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005',
         '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
         '2015', '2016', '2017', '2018'] #, '2019']  #'1979'

bado_yr_years = ['1981', '1982', '1983', '1984', '1985', '1986', '1987',
         '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996',
         '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005',
         '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
         '2015']  # '1980', 

bado_day_years = ['1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987',
         '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996',
         '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005',
         '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
         '2015']

cmip6_years = {'historical': ['1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987',
         '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996',
         '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005',
         '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014'],
               'amip': ['1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987',
         '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996',
         '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005',
         '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014'],
               'ssp': ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022',
         '2023', '2024', '2025', '2026', '2027', '2028', '2029', '2030', '2031', '2032',
         '2033', '2034', '2035', '2036', '2037', '2038', '2039', '2040', '2041', '2042',
         '2043', '2044', '2045', '2046', '2047', '2048', '2049', '2050', '2051', '2052',
         '2053', '2054', '2055', '2056', '2057', '2058', '2059', '2060', '2061', '2062',
         '2063', '2064', '2065', '2066', '2067', '2068', '2069', '2070', '2071', '2072',
         '2073', '2074', '2075', '2076', '2077', '2078', '2079', '2080', '2081', '2082',
         '2083', '2084', '2085', '2086', '2087', '2088', '2089', '2090', '2091', '2092',
         '2093', '2094', '2095', '2096', '2097', '2098', '2099', '2100']}

#cmip6_years_picontrol

chirps_years = ['1981', '1982', '1983', '1984', '1985', '1986', '1987',
         '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996',
         '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005',
         '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
         '2015', '2016', '2017', '2018']

sat_years = {'CHIRPS': chirps_years}


#~ Spatial coverage

lat_min = 0.
lat_max = 20.
lon_min = -20.
lon_max = 20.

lat_min_wind = 0.
lat_max_wind = 30.
lon_min_wind = -20.
lon_max_wind = 20.


# CMIP6 precip limit coords:

lats_min = {'CMIP6': -90., 'CMIP6_SAHEL': 0., 'CMIP6_EU': 35., 'CMIP6_GR': 60., 'CMIP5': 0.}
lats_max = {'CMIP6': 90., 'CMIP6_SAHEL': 20., 'CMIP6_EU': 60., 'CMIP6_GR': 85., 'CMIP5': 20.}
lons_min = {'CMIP6': -180., 'CMIP6_SAHEL': -20., 'CMIP6_EU': -10., 'CMIP6_GR': -70., 'CMIP5': -20.}
lons_max = {'CMIP6': 180., 'CMIP6_SAHEL': 20., 'CMIP6_EU': 25., 'CMIP6_GR': -10., 'CMIP5': 20.}


"""lats_min = {'CMIP6': {'IPSL-CM6A-LR': 0., 'CanESM5': 1.39530691, 'CESM2': 0.47120419, 'CNRM-CM6-1': 0.7003838, 'MPI-ESM1-2-LR': 0.93262997, 'MIROC6': 0.7003838, 'EC-Earth3': 0.3508765, 'ACCESS-ESM1-5': 0., 'UKESM1-0-LL': 0.625, 'INM-CM5-0': 0.75}, 'CMIP6_EU': {'MPI-ESM1-2-LR': 35.}}
lats_max = {'CMIP6': {'IPSL-CM6A-LR': 19.014084, 'CanESM5': 18.13897099, 'CESM2': 19.31937173, 'CNRM-CM6-1': 18.91035725, 'MPI-ESM1-2-LR': 19.58521861, 'MIROC6': 18.91035725, 'EC-Earth3': 19.99996038, 'ACCESS-ESM1-5': 20., 'UKESM1-0-LL': 19.375, 'INM-CM5-0': 18.75}, 'CMIP6_EU': {'MPI-ESM1-2-LR': 60.}}
lons_min = {'CMIP6': {'IPSL-CM6A-LR': -20., 'CanESM5': -19.6875, 'CESM2': -20, 'CNRM-CM6-1': -19.6875, 'MPI-ESM1-2-LR': -18.75, 'MIROC6': -19.6875, 'EC-Earth3': -19.6875, 'ACCESS-ESM1-5': -18.75, 'UKESM1-0-LL': -19.6875, 'INM-CM5-0': -20.}, 'CMIP6_EU': {'MPI-ESM1-2-LR': -10.}}
lons_max = {'CMIP6': {'IPSL-CM6A-LR': 20., 'CanESM5': 19.6875, 'CESM2': 20, 'CNRM-CM6-1': 19.6875, 'MPI-ESM1-2-LR': 18.75, 'MIROC6': 19.6875, 'EC-Earth3': 19.6875, 'ACCESS-ESM1-5': 18.75, 'UKESM1-0-LL': 19.6875, 'INM-CM5-0': 20.}, 'CMIP6_EU': {'MPI-ESM1-2-LR': 25.}}"""


# GCM resolution as a function of CMIP and experiment

lat_res = {'CMIP6' : {'historical': {'ACCESS-ESM1-5': 1.25, 'AWI-ESM-1-1-LR': 1.86, 'BCC-CSM2-MR': 1.12, 'BCC-ESM1': 2.79, 'CanESM5': 2.79, 'CESM2': 0.94, 'CMCC-ESM2': 0.94, 'CNRM-CM6-1': 1.4, 'CNRM-ESM2-1': 1.4, 'EC-Earth3': 0.35, 'FGOALS-f3-L': 1., 'FGOALS-g3': 2.28, 'GFDL-CM4': 1., 'GFDL-ESM4': 1., 'HadGEM3-GC31-LL': 1.25, 'INM-CM5-0': 1.5, 'IPSL-CM6A-LR': 1.27, 'MIROC6': 2.79, 'MIROC6': 1.4, 'MPI-ESM1-2-LR': 1.86, 'MRI-ESM2-0': 1.12, 'NorESM2-LM': 1.89, 'SAM0-UNICON': 0.94, 'EC-Earth3': 0.35, 'UKESM1-0-LL': 1.25}, 
           'piControl': {'ACCESS-ESM1-5': 1.25, 'BCC-CSM2-MR': 1.12, 'CESM2': 0.94, 'CNRM-CM6-1': 1.4, 'HadGEM3-GC31-LL': 1.25, 'INM-CM5-0': 1.5, 'IPSL-CM6A-LR': 1.27, 'MPI-ESM1-2-LR': 1.86, 'MIROC6': 1.4, 'MRI-ESM2-0': 1.12, 'NorESM2-LM': 1.89, 'UKESM1-0-LL': 1.25}, 
           'ssp126': {'CanESM5': 2.79, 'IPSL-CM6A-LR': 1.27, 'BCC-CSM2-MR': 1.12, 'CESM2': 0.94, 'CNRM-CM6-1': 1.4, 'MPI-ESM1-2-LR': 1.86, 'MIROC6': 1.4, 'ACCESS-ESM1-5': 1.25, 'UKESM1-0-LL': 1.25, 'INM-CM5-0': 1.5},
           'ssp245': {'CanESM5': 2.79, 'IPSL-CM6A-LR': 1.27, 'BCC-CSM2-MR': 1.12, 'CESM2': 0.94, 'CNRM-CM6-1': 1.4, 'EC-Earth3': 0.35, 'MPI-ESM1-2-LR': 1.86, 'MIROC6': 1.4, 'ACCESS-ESM1-5': 1.25, 'UKESM1-0-LL': 1.25, 'INM-CM5-0': 1.5},
           'ssp585': {'CanESM5': 2.79, 'IPSL-CM6A-LR': 1.27, 'BCC-CSM2-MR': 1.12, 'CESM2': 0.94, 'CNRM-CM6-1': 1.4, 'MPI-ESM1-2-LR': 1.86, 'MIROC6': 1.4, 'ACCESS-ESM1-5': 1.25, 'UKESM1-0-LL': 1.25, 'INM-CM5-0': 1.5},
           '1pctCO2': {'ACCESS-ESM1-5': 1.25, 'CESM2': 0.94, 'CNRM-CM6-1': 1.4, 'GFDL-CM4': 1., 'HadGEM3-GC31-LL': 1.25, 'IPSL-CM6A-LR': 1.27, 'INM-CM5-0': 1.5, 'MIROC6': 1.4, 'MPI-ESM1-2-LR': 1.86, 'MRI-ESM2-0': 1.12, 'NorESM2-LM': 1.89, 'UKESM1-0-LL': 1.25},
           'abrupt-4xCO2': {'ACCESS-ESM1-5': 1.25, 'CESM2': 0.94, 'CNRM-CM6-1': 1.4, 'GFDL-CM4': 1., 'IPSL-CM6A-LR': 1.27, 'INM-CM5-0': 1.5, 'MIROC6': 1.4, 'MPI-ESM1-2-LR': 1.86, 'MRI-ESM2-0': 1.12, 'UKESM1-0-LL': 1.25},
           'amip-4xCO2': {'GFDL-CM4': 1.},
           'amip': {'ACCESS-ESM1-5': 1.25, 'BCC-CSM2-MR': 1.12, 'CESM2': 0.94, 'CNRM-CM6-1': 1.4, 'GFDL-CM4': 1., 'HadGEM3-GC31-LL': 1.25, 'INM-CM5-0': 1.5, 'IPSL-CM6A-LR': 1.27, 'MIROC6': 1.4, 'MPI-ESM1-2-LR': 1.86, 'MRI-ESM2-0': 1.12, 'NorESM2-LM': 1.89, 'UKESM1-0-LL': 1.25},
           'hist-GHG': {'ACCESS-ESM1-5': 1.25, 'BCC-CSM2-MR': 1.12, 'CESM2': 0.94, 'CNRM-CM6-1': 1.4, 'HadGEM3-GC31-LL': 1.25, 'IPSL-CM6A-LR': 1.27, 'MIROC6': 1.4, 'MRI-ESM2-0': 1.12, 'NorESM2-LM': 1.89},
           'hist-aer': {'ACCESS-ESM1-5': 1.25, 'BCC-CSM2-MR': 1.12, 'CESM2': 0.94, 'CNRM-CM6-1': 1.4, 'HadGEM3-GC31-LL': 1.25, 'IPSL-CM6A-LR': 1.27, 'MIROC6': 1.4, 'MRI-ESM2-0': 1.12, 'NorESM2-LM': 1.89},
           'hist-nat': {'ACCESS-ESM1-5': 1.25, 'BCC-CSM2-MR': 1.12, 'CESM2': 0.94, 'CNRM-CM6-1': 1.4, 'HadGEM3-GC31-LL': 1.25, 'IPSL-CM6A-LR': 1.27, 'MIROC6': 1.4, 'MRI-ESM2-0': 1.12, 'NorESM2-LM': 1.89}},
          'CMIP5': {'historical': {'ACCESS1-3': 1.25, 'CESM1-CAM5': 0.94, 'CNRM-CM5': 1.4, 'CSIRO-Mk3-6-0': 1.86, 'GFDL-CM3': 2., 'IPSL-CM5A-LR': 1.89, 'MPI-ESM-LR': 1.86, 'NorESM1-M': 1.89}}}

lon_res = {'CMIP6': {'historical': {'ACCESS-ESM1-5': 1.88, 'AWI-ESM-1-1-LR': 1.88, 'BCC-CSM2-MR': 1.12, 'BCC-ESM1': 2.81, 'CanESM5': 2.81, 'CESM2': 1.25, 'CMCC-ESM2': 1.25, 'CNRM-CM6-1': 1.41, 'CNRM-ESM2-1': 1.41, 'EC-Earth3': 0.7, 'FGOALS-f3-L': 1.25, 'FGOALS-g3': 2., 'GFDL-CM4': 1.25, 'GFDL-ESM4': 1.25, 'HadGEM3-GC31-LL': 1.88, 'INM-CM5-0': 2., 'IPSL-CM6A-LR': 2.5, 'MIROC-ES2L': 2.81, 'MIROC6': 1.41, 'MPI-ESM1-2-LR': 1.88, 'MRI-ESM2-0': 1.12, 'NorESM2-LM': 2.5, 'SAM0-UNICON': 1.25, 'EC-Earth3': 0.7, 'UKESM1-0-LL': 1.88},
           'piControl': {'ACCESS-ESM1-5': 1.88, 'BCC-CSM2-MR': 1.12, 'CESM2': 1.25, 'CNRM-CM6-1': 1.41, 'HadGEM3-GC31-LL': 1.88, 'INM-CM5-0': 2., 'IPSL-CM6A-LR': 2.5, 'MPI-ESM1-2-LR': 1.88, 'MIROC6': 1.41, 'MRI-ESM2-0': 1.12, 'NorESM2-LM': 2.5, 'UKESM1-0-LL': 1.88},
           'ssp126': {'CanESM5': 2.81, 'IPSL-CM6A-LR': 2.5, 'BCC-CSM2-MR': 1.12, 'CESM2': 1.25, 'CNRM-CM6-1': 1.41, 'MPI-ESM1-2-LR': 1.88, 'MIROC6': 1.41, 'ACCESS-ESM1-5': 1.88, 'UKESM1-0-LL': 1.88, 'INM-CM5-0': 2.},
           'ssp245': {'CanESM5': 2.81, 'IPSL-CM6A-LR': 2.5, 'BCC-CSM2-MR': 1.12, 'CESM2': 1.25, 'CNRM-CM6-1': 1.41, 'EC-Earth3': 0.7, 'MPI-ESM1-2-LR': 1.88, 'MIROC6': 1.41, 'ACCESS-ESM1-5': 1.88, 'UKESM1-0-LL': 1.88, 'INM-CM5-0': 2.},
           'ssp585': {'CanESM5': 2.81, 'IPSL-CM6A-LR': 2.5, 'BCC-CSM2-MR': 1.12, 'CESM2': 1.25, 'CNRM-CM6-1': 1.41, 'MPI-ESM1-2-LR': 1.88, 'MIROC6': 1.41, 'ACCESS-ESM1-5': 1.88, 'UKESM1-0-LL': 1.88, 'INM-CM5-0': 2.},
           '1pctCO2': {'ACCESS-ESM1-5': 1.88, 'CESM2': 1.25, 'CNRM-CM6-1': 1.41, 'GFDL-CM4': 1.25, 'HadGEM3-GC31-LL': 1.88,  'IPSL-CM6A-LR': 2.5, 'INM-CM5-0': 2., 'MIROC6': 1.41, 'MPI-ESM1-2-LR': 1.88, 'MRI-ESM2-0': 1.12, 'NorESM2-LM': 2.5, 'UKESM1-0-LL': 1.88},
           'abrupt-4xCO2': {'ACCESS-ESM1-5': 1.88, 'CESM2': 1.25, 'CNRM-CM6-1': 1.41, 'GFDL-CM4': 1.25, 'IPSL-CM6A-LR': 2.5, 'INM-CM5-0': 2., 'MIROC6': 1.41, 'MPI-ESM1-2-LR': 1.88, 'MRI-ESM2-0': 1.12, 'UKESM1-0-LL': 1.88},
           'amip-4xCO2': {'GFDL-CM4': 1.25},
           'amip': {'ACCESS-ESM1-5': 1.88, 'BCC-CSM2-MR': 1.12, 'CESM2': 1.25, 'CNRM-CM6-1': 1.41, 'GFDL-CM4': 1.25, 'HadGEM3-GC31-LL': 1.88, 'INM-CM5-0': 2., 'IPSL-CM6A-LR': 2.5, 'MIROC6': 1.41, 'MPI-ESM1-2-LR': 1.88, 'MRI-ESM2-0': 1.12, 'NorESM2-LM': 2.5, 'UKESM1-0-LL': 1.88},
           'hist-GHG': {'ACCESS-ESM1-5': 1.88, 'BCC-CSM2-MR': 1.12, 'CESM2': 1.25, 'CNRM-CM6-1': 1.41, 'HadGEM3-GC31-LL': 1.88, 'IPSL-CM6A-LR': 2.5, 'MIROC6': 1.41, 'MRI-ESM2-0': 1.12, 'NorESM2-LM': 2.5},
           'hist-aer': {'ACCESS-ESM1-5': 1.88, 'BCC-CSM2-MR': 1.12, 'CESM2': 1.25, 'CNRM-CM6-1': 1.41, 'HadGEM3-GC31-LL': 1.88, 'IPSL-CM6A-LR': 2.5, 'MIROC6': 1.41, 'MRI-ESM2-0': 1.12, 'NorESM2-LM': 2.5},
           'hist-nat': {'ACCESS-ESM1-5': 1.88, 'BCC-CSM2-MR': 1.12, 'CESM2': 1.25, 'CNRM-CM6-1': 1.41, 'HadGEM3-GC31-LL': 1.88, 'IPSL-CM6A-LR': 2.5, 'MIROC6': 1.41, 'MRI-ESM2-0': 1.12, 'NorESM2-LM': 2.5}},
           'CMIP5': {'historical': {'ACCESS1-3': 1.88, 'CESM1-CAM5': 1.25, 'CNRM-CM5': 1.41, 'CSIRO-Mk3-6-0': 1.88, 'GFDL-CM3': 2.5, 'IPSL-CM5A-LR': 3.75, 'MPI-ESM-LR': 1.88, 'NorESM1-M': 2.5}}}





plvls = ['200', '500', '600', '700', '800', '850', '925']

g = 9.81

#~ Covariates dates
# T2M / D2M
start_date_t2m = [7, 1, 0]
end_date_t2m = [9, 30, 23]

# TCW
start_date_tcw_monthly = 7  # month
end_date_tcw_monthly = 9    # month
#h=18


#~ ERA5 variables format

# t2m, d2m:          00:00 -> 23:00            / 01/01 -> 31/12 / 0:20, -20:20 /                                   / K
# tcw:               00:00, 6:00, 12:00, 18:00 / 01/01 -> 31/12 / 0:20, -20:20 /                                   / kg/m2
# tcw monthly mean:  18:00                     / 01/01 -> 31/12 / 0:20, -20:20 /                                   / kg/m2
# q :                00:00, 6:00, 12:00, 18:00 / 01/06 -> 30/09 / 0:20, -20:20 / 925, 850, 800, 700, 600, 500, 250 / kg/kg
# wind (u, v):       00:00, 6:00, 12:00, 18:00 / 01/06 -> 30/09 / 0:30, -20:20 / 925, 850, 800, 700, 600, 500, 250 / m/s
# tp :               00:00 -> 23:00            / 01/01 -> 31/12 / 0:20, -20:20 /                                   / m
# sst monthly mean:                            / 01/01 -> 31/12 /              /                                   / K




