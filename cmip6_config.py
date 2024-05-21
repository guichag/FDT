"""Config file for cmip6 scripts"""

# Expeirment color plot
exp_cols = {'historical': 'k', 'hist-nat': 'c', 'hist-aer': 'g', 'hist-GHG': 'purple', 'ssp245': 'b', 'ssp585': 'm'}


# Wet day threshold (mm/d)

TH_WD = {'ACCESS-ESM1-5': 4., 'AWI-ESM-1-1-LR': 6., 'BCC-ESM1': 1., 'BCC-CSM2-MR': 1.5, 'CESM2': 4., 'CMCC-ESM2': 4., 'CNRM-CM6-1': 3.5, 'FGOALS-f3-L': 0.4, 'FGOALS-g3': 1.5, 'GFDL-CM4': 3.5, 'GFDL-ESM4': 3.5, 'HadGEM3-GC31-LL': 2.5, 'INM-CM5-0': 2.5, 'IPSL-CM6A-LR': 6., 'MIROC-ES2L': 4.5, 'MIROC6': 4.5, 'MPI-ESM1-2-LR': 5., 'MRI-ESM2-0': 4.5, 'NorESM2-LM': 4., 'SAM0-UNICON': 7., 'UKESM1-0-LL': 2.5}  # -> CONSIDERING THE WHOLE YEAR: OK !!!

#TH_WD = {'ACCESS-ESM1-5': 2., 'BCC-CSM2-MR': 0.75, 'CESM2': 2., 'CNRM-CM6-1': 1.75, 'HadGEM3-GC31-LL': 1.25, 'INM-CM5-0': 1.25, 'IPSL-CM6A-LR': 3., 'MIROC6': 2.25, 'MPI-ESM1-2-LR': 2.5, 'MRI-ESM2-0': 2.25, 'NorESM2-LM': 2., 'UKESM1-0-LL': 1.25}  # -> ORIGINAL THRESHOLD DIVIDED BY 2 ( CONSIDERING THE WHOLE YEAR: OK !!!)

#TH_WD = {'ACCESS-ESM1-5': 1.33, 'BCC-CSM2-MR': 0.5, 'CESM2': 1.33, 'CNRM-CM6-1': 1.17, 'HadGEM3-GC31-LL': 0.83, 'INM-CM5-0': 0.83, 'IPSL-CM6A-LR': 2., 'MIROC6': 1.5, 'MPI-ESM1-2-LR': 1.67, 'MRI-ESM2-0': 1.5, 'NorESM2-LM': 1.33, 'UKESM1-0-LL': 0.83}  # -> ORIGINAL THRESHOLD DIVIDED BY 3 ( CONSIDERING THE WHOLE YEAR: OK !!!)

#TH_WD = {'ACCESS-ESM1-5': 3.5, 'CESM2': 3.75, 'CNRM-CM6-1': 3.25, 'HadGEM3-GC31-LL': 2., 'IPSL-CM6A-LR': 6., 'MIROC6': 4.5, 'MRI-ESM2-0': 4.5, 'NorESM2-LM': 4.}  # -> CONSIDERING JJAS ONLY


# Dates

ymin_obs = 1950
ymax_obs = 2014
ymin_hist = 1850
ymax_hist = 2014
ymin_ssp = 2015
ymax_ssp = 2100
