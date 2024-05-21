"""NS GEV"""

import numpy as np
import pandas as pd

from stats import VarDistribution, CovarDistribution
from stats.math_functions import LinearExpr
from stats.math_exprs import MathExpr, Linear_Combination


### FUNC ###

#------------------------------------------ Normalized covariate(s) ----------------------------------------------#

def get_vdistr_norm(model, params, covar):
    """generate a variable distribution"""
    assert isinstance(params, dict), 'parameters must be stored in dict'

    loc = params['loc']
    scale = params['scale']
    c = params['c']

    assert len(loc) == 2 or len(scale) == 2, 'at least one of the parameters must be [slope, intercept] shaped for NS distribution'

    if len(loc) == 2:
        loc_sl = loc[0]
        loc_int = loc[1]
        loc_expr = LinearExpr([loc_sl, loc_int])
    else:
        loc_expr = loc[0]

    if len(scale) == 2:
        scale_sl = scale[0]
        scale_int = scale[1]
        scale_expr = LinearExpr([scale_sl, scale_int])
    else:
        scale_expr = scale[0]

    # normalize covariate
    covar_norm = (covar - np.nanmin(covar)) / (np.nanmax(covar) - np.nanmin(covar))

    inds = np.arange(0, len(covar))

    vdistr = VarDistribution(model, loc=loc_expr, scale=scale_expr, c=c)

    x = pd.DataFrame({"x": covar_norm})
    x.index = covar_norm

    output = vdistr(x)

    output.index = inds

    return output


def get_vdistr_norm_disp(model, params, covar):
    """generate a variable distribution for disp model"""
    assert isinstance(params, dict), 'parameters must be stored in dict'

    loc = params['loc']
    scale = params['scale']
    c = params['c']

    assert len(loc) == 2 or len(scale) == 2, 'at least one of the parameters must be [slope, intercept] shaped for NS distribution'

    loc_sl = loc[0]
    loc_int = loc[1]
    loc_expr = Linear_Combination({'intercept': loc_int, 't': loc_sl})

    scale_int = scale[0]
    scale_expr = Linear_Combination({'intercept': scale_int, 't': loc_sl*scale_int/loc_int})

    # normalize covariate
    covar_norm = (covar - np.nanmin(covar)) / (np.nanmax(covar) - np.nanmin(covar))

    inds = np.arange(0, len(covar))

    vdistr = CovarDistribution(model, loc=loc_expr, scale=scale_expr, c=c)

    x = pd.DataFrame({"t": covar_norm})
    x.index = covar_norm

    output = vdistr(x)

    output.index = inds

    return output


def get_vdistr_norm_2(model, params, covar):
    """generate a variable distribution with covar in [-0.5:0.5]"""
    assert isinstance(params, dict), 'parameters must be stored in dict'

    loc = params['loc']
    scale = params['scale']
    c = params['c']

    assert len(loc) == 2 or len(scale) == 2, 'at least one of the parameters must be [slope, intercept] shaped for NS distribution'

    if len(loc) == 2:
        loc_sl = loc[0]
        loc_int = loc[1]
        #loc_expr = Linear_Combination({'intercept': loc_int, 't': loc_sl})  # LinearExpr([loc_sl, loc_int])
        # FOR TEST: mu(s,t) = mu(s)(1+mu3*t)
        loc_expr = MathExpr('{0}*(1+{1}*t)'.format(loc_int, loc_sl))
    else:
        loc_expr = loc[0]

    if len(scale) == 2:
        scale_sl = scale[0]
        scale_int = scale[1]
        #scale_expr = Linear_Combination({'intercept': scale_int, 't': scale_sl})  # LinearExpr([loc_sl, loc_int])
        # FOR TEST: scale(s,t) = scale(s)*(1+scale3*t)
        scale_expr = MathExpr('{0}*(1+{1}*t)'.format(scale_int, scale_sl))

    else:
        scale_expr = scale[0]


    # normalize covariate
    covar_norm = (covar - np.nanmin(covar)) / (np.nanmax(covar) - np.nanmin(covar)) - 0.5  # FOR ST-NS-GEV MODELS

    inds = np.arange(0, len(covar))

    vdistr = CovarDistribution(model, loc=loc_expr, scale=scale_expr, c=c)

    x = pd.DataFrame({"t": covar_norm})
    x.index = covar_norm

    output = vdistr(x)

    output.index = inds

    return output


def get_vdistr_norm_disp_2(model, params, covar):
    """generate a variable distribution for disp model with covar in [-0.5:0.5]"""
    assert isinstance(params, dict), 'parameters must be stored in dict'

    loc = params['loc']
    scale = params['scale']
    c = params['c']

    assert len(loc) == 2 or len(scale) == 2, 'at least one of the parameters must be [slope, intercept] shaped for NS distribution'

    loc_sl = loc[0]
    loc_int = loc[1]
    #loc_expr = Linear_Combination({'intercept': loc_int, 't': loc_sl})
    # FOR TEST: mu(s,t) = mu(s)(1+mu3*t)
    loc_expr = MathExpr('{0}*(1+{1}*t)'.format(loc_int, loc_sl))

    scale_int = scale[0]
    #scale_expr = Linear_Combination({'intercept': scale_int, 't': loc_sl*scale_int/loc_int})
    # FOR TEST: scale(s,t) = scale(s)*(1+scale3*t)
    scale_expr = MathExpr('{0}*(1+{1}*t)'.format(scale_int, loc_sl))


    # normalize covariate
    covar_norm = (covar - np.nanmin(covar)) / (np.nanmax(covar) - np.nanmin(covar)) - 0.5  # FOR ST-NS-GEV MODELS

    inds = np.arange(0, len(covar))

    vdistr = CovarDistribution(model, loc=loc_expr, scale=scale_expr, c=c)

    x = pd.DataFrame({"t": covar_norm})
    x.index = covar_norm

    output = vdistr(x)

    output.index = inds

    return output  # loc_expr, scale_expr, c


#-------------------------------------- For full Spatially NS-GEV models ----------------------------------------#

def get_vdistr_norm_sns_full(model, params, covars_space):
    """generate a variable distribution with time- and space-dependent parameters"""
    assert isinstance(params, dict), 'parameters must be stored in dict'

    loc = params['loc']
    scale = params['scale']
    c = params['c']

    covar_lat = covars_space['lats'].values
    covar_lon = covars_space['lons'].values

    assert len(loc) == 3 or len(scale) == 3, 'at least one of the parameters must be [lat_slope, lon_slope, intercept] shaped for full SNS-GEV model fit'
    # -> add the case where one parameter is not space-dependent

    loc_lat_sl = loc[0]
    loc_lon_sl = loc[1]
    loc_int = loc[2]
    loc_expr = Linear_Combination({'intercept': loc_int, 'lat': loc_lat_sl, 'lon': loc_lon_sl})

    scale_lat_sl = scale[0]
    scale_lon_sl = scale[1]
    scale_int = scale[2]
    scale_expr = Linear_Combination({'intercept': scale_int, 'lat': scale_lat_sl, 'lon': scale_lon_sl})

    # normalize covariates
    covar_lat_norm = (covar_lat - np.nanmin(covar_lat)) / (np.nanmax(covar_lat) - np.nanmin(covar_lat))
    covar_lon_norm = (covar_lon - np.nanmin(covar_lon)) / (np.nanmax(covar_lon) - np.nanmin(covar_lon))

    covdistr = CovarDistribution(model, loc=loc_expr, scale=scale_expr, c=c)

    x = pd.DataFrame({"lat": covar_lat_norm, "lon": covar_lon_norm})

    output = covdistr(x)

    return output


#--------------------------------- For Spatially and Temporally NS-GEV models ------------------------------------#

def get_vdistr_norm_st_BAK(model, params, covar_time, covar_space):
    """generate a variable distribution with time- and space-dependent parameters"""
    assert isinstance(params, dict), 'parameters must be stored in dict'

    loc = params['loc']
    scale = params['scale']
    c = params['c']

    assert len(loc) == 3 or len(scale) == 3, 'at least one of the parameters must be [time_slope, space_slope, intercept] shaped for NS distribution' 
    # -> add the case where one parameter is not space-dependent

    if len(loc) == 2:       # space-dependent only location parameter
        loc_s_sl = loc[0]
        loc_s_int = loc[1]
        loc_expr = MathExpr('{0}*s + {1}'.format(loc_s_sl, loc_s_int))

    elif len(loc) == 3:     # space- and time-dependent location parameter
        loc_t_sl = loc[0]
        loc_s_sl = loc[1]
        loc_int = loc[2]
        loc_expr = Linear_Combination({'intercept': loc_int, 't': loc_t_sl, 's': loc_s_sl})

    if len(scale) == 2:       # space-dependent only scale parameter
        scale_s_sl = scale[0]
        scale_s_int = scale[1]
        scale_expr = MathExpr('{0}*s + {1}'.format(scale_s_sl, scale_s_int))

    elif len(scale) == 3:     # space- and time-dependent scale parameter
        scale_t_sl = scale[0]
        scale_s_sl = scale[1]
        scale_int = scale[2]
        scale_expr = Linear_Combination({'intercept': scale_int, 't': scale_t_sl, 's': scale_s_sl})

    # normalize covariates
    covar_time_norm = (covar_time - np.nanmin(covar_time)) / (np.nanmax(covar_time) - np.nanmin(covar_time))
    covar_space_norm = (covar_space - np.nanmin(covar_space)) / (np.nanmax(covar_space) - np.nanmin(covar_space))

    covdistr = CovarDistribution(model, loc=loc_expr, scale=scale_expr, c=c)

    x = pd.DataFrame({"t": covar_time_norm, "s": covar_space_norm})

    output = covdistr(x)

    return output


def get_vdistr_norm_st(model, params, covar_time, covar_space):
    """generate a variable distribution with time- and space-dependent parameters"""
    assert isinstance(params, dict), 'parameters must be stored in dict'

    loc = params['loc']
    scale = params['scale']
    c = params['c']

    assert len(loc) == 3 or len(scale) == 3, 'at least one of the parameters must be [time_slope, space_slope, intercept] shaped for ST-NS-GEV model fit'
    # -> add the case where one parameter is not space-dependent

    if len(loc) == 2:       # space-dependent only location parameter
        loc_s_sl = loc[0]
        loc_s_int = loc[1]
        loc_expr = MathExpr('{0}*s + {1}'.format(loc_s_sl, loc_s_int))

    elif len(loc) == 3:     # space- and time-dependent location parameter
        loc_t_sl = loc[0]
        loc_s_sl = loc[1]
        loc_int = loc[2]
        #loc_expr = Linear_Combination({'intercept': loc_int, 't': loc_t_sl, 's': loc_s_sl})
        # FOR TEST: mu(s,t) = mu(s)(1+mu2*t)
        loc_expr = MathExpr('({0} + {1}*s)*(1 + {2}*t)'.format(loc_int, loc_s_sl, loc_t_sl))

    if len(scale) == 2:       # space-dependent only scale parameter
        scale_s_sl = scale[0]
        scale_s_int = scale[1]
        scale_expr = MathExpr('{0}*s + {1}'.format(scale_s_sl, scale_s_int))

    elif len(scale) == 3:     # space- and time-dependent scale parameter
        scale_t_sl = scale[0]
        scale_s_sl = scale[1]
        scale_int = scale[2]
        #scale_expr = Linear_Combination({'intercept': scale_int, 't': scale_t_sl, 's': scale_s_sl})
        # FOR TEST* sigma(s,t) = sigma(s)*(1+sigma2*t)
        scale_expr = MathExpr('({0} + {1}*s)*(1 + {2}*t)'.format(scale_int, scale_s_sl, scale_t_sl))

    # normalize covariates
    covar_time_norm = (covar_time - np.nanmin(covar_time)) / (np.nanmax(covar_time) - np.nanmin(covar_time)) - 0.5
    covar_space_norm = (covar_space - np.nanmin(covar_space)) / (np.nanmax(covar_space) - np.nanmin(covar_space))

    covdistr = CovarDistribution(model, loc=loc_expr, scale=scale_expr, c=c)

    x = pd.DataFrame({"t": covar_time_norm, "s": covar_space_norm})

    output = covdistr(x)

    return output


#------------------- For Spatially and Temporally NS-GEV model with constant sigma/mu ratio ---------------------#

def get_vdistr_disp_norm_st(model, params, covar_time, covar_space):
    """generate a variable distribution with time- and space-dependent parameters for constant sigma/mu models"""
    assert isinstance(params, dict), 'parameters must be stored in dict'

    loc = params['loc']
    scale = params['scale']
    c = params['c']

    assert len(loc) == 3 or len(scale) == 3, 'at least one of the parameters must be [time_slope, space_slope, intercept] shaped for ST-NS-GEV model fit'
    # -> add the case where one parameter is not space-dependent

    if len(loc) == 2:       # space-dependent only location parameter
        loc_s_sl = loc[0]
        loc_s_int = loc[1]
        loc_expr = MathExpr('{0}*s + {1}'.format(loc_s_sl, loc_s_int))

    elif len(loc) == 3:     # space- and time-dependent location parameter
        loc_t_sl = loc[0]
        loc_s_sl = loc[1]
        loc_int = loc[2]
        #loc_expr = Linear_Combination({'intercept': loc_int, 't': loc_t_sl, 's': loc_s_sl})
        # FOR TEST: mu(s,t) = mu(s)(1+mu3*t)
        loc_expr = MathExpr('({0} + {1}*s)*(1 + {2}*t)'.format(loc_int, loc_s_sl, loc_t_sl))

    scale_s_sl = scale[0]
    scale_int = scale[1]

    #scale_expr = MathExpr('{0} + {1}*s + {2}*({0}+{1}*s) / ({3}+{4}*s) * t'.format(scale_int, scale_s_sl, loc_t_sl, loc_int, loc_s_sl))
    scale_expr = MathExpr('({0} + {1}*s)*(1 + {2}*t)'.format(scale_int, scale_s_sl, loc_t_sl))


    # normalize covariates
    covar_time_norm = (covar_time - np.nanmin(covar_time)) / (np.nanmax(covar_time) - np.nanmin(covar_time)) - 0.5
    covar_space_norm = (covar_space - np.nanmin(covar_space)) / (np.nanmax(covar_space) - np.nanmin(covar_space))

    covdistr = CovarDistribution(model, loc=loc_expr, scale=scale_expr, c=c)

    x = pd.DataFrame({"t": covar_time_norm, "s": covar_space_norm})

    output = covdistr(x)

    return output


#--------------------- For Spatially and Temporally NS-GEV models fitted on station data -----------------------#

def get_vdistr_norm_st_point(model, params, covar_time, covar_space_norm):
    """generate a variable distribution with time- and space-dependent parameters for point (station/grid cell) data"""
    assert isinstance(params, dict), 'parameters must be stored in dict'

    loc = params['loc']
    scale = params['scale']
    c = params['c']

    assert len(loc) == 3 or len(scale) == 3, 'at least one of the parameters must be [time_slope, space_slope, intercept] shaped for NS distribution' 
    # -> add the case where one parameter is not space-dependent

    if len(loc) == 2:       # space-dependent only location parameter
        loc_s_sl = loc[0]
        loc_s_int = loc[1]
        loc_expr = MathExpr('{0}*s + {1}'.format(loc_s_sl, loc_s_int))

    elif len(loc) == 3:     # space- and time-dependent location parameter
        loc_t_sl = loc[0]
        loc_s_sl = loc[1]
        loc_int = loc[2]
        loc_expr = Linear_Combination({'intercept': loc_int, 't': loc_t_sl, 's': loc_s_sl})

    if len(scale) == 2:       # space-dependent only scale parameter
        scale_s_sl = scale[0]
        scale_s_int = scale[1]
        scale_expr = MathExpr('{0}*s + {1}'.format(scale_s_sl, scale_s_int))

    elif len(scale) == 3:     # space- and time-dependent scale parameter
        scale_t_sl = scale[0]
        scale_s_sl = scale[1]
        scale_int = scale[2]
        scale_expr = Linear_Combination({'intercept': scale_int, 't': scale_t_sl, 's': scale_s_sl})

    # normalize covariate (only time, space covar already normalized)
    covar_time_norm = (covar_time - np.nanmin(covar_time)) / (np.nanmax(covar_time) - np.nanmin(covar_time)) - 0.5
    #covar_space_norm = (covar_space - np.nanmin(covar_space)) / (np.nanmax(covar_space) - np.nanmin(covar_space))

    covdistr = CovarDistribution(model, loc=loc_expr, scale=scale_expr, c=c)

    x = pd.DataFrame({"t": covar_time_norm, "s": covar_space_norm})

    output = covdistr(x)

    return output


#-------------------- For Spatially and Temporally NS-GEV models fitted all at once (USELESS) --------------------#

def get_vdistr_norm_st_all(model, params, covar_time, covar_space):
    """generate a variable distribution with time- and space-dependent parameters"""
    assert isinstance(params, dict), 'parameters must be stored in dict'

    loc = params['loc']
    scale = params['scale']
    c = params['c']

    assert len(loc) == 3 or len(scale) == 3, 'at least one of the parameters must be [time_slope, space_slope, intercept] shaped for ST-NS-GEV model fit'
    # -> add the case where one parameter is not space-dependent

    if len(loc) == 2:       # space-dependent only location parameter
        loc_s_sl = loc[0]
        loc_s_int = loc[1]
        loc_expr = MathExpr('{0}*s + {1}'.format(loc_s_sl, loc_s_int))

    elif len(loc) == 3:     # space- and time-dependent location parameter
        loc_t_sl = loc[0]
        loc_s_sl = loc[1]
        loc_int = loc[2]
        loc_expr = Linear_Combination({'intercept': loc_int, 't': loc_t_sl, 's': loc_s_sl})

    if len(scale) == 2:       # space-dependent only scale parameter
        scale_s_sl = scale[0]
        scale_s_int = scale[1]
        scale_expr = MathExpr('{0}*s + {1}'.format(scale_s_sl, scale_s_int))

    elif len(scale) == 3:     # space- and time-dependent scale parameter
        scale_t_sl = scale[0]
        scale_s_sl = scale[1]
        scale_int = scale[2]
        scale_expr = Linear_Combination({'intercept': scale_int, 't': scale_t_sl, 's': scale_s_sl})

    # normalize covariates
    covar_time_norm = (covar_time - np.nanmin(covar_time)) / (np.nanmax(covar_time) - np.nanmin(covar_time)) - 0.5
    covar_space_norm = (covar_space - np.nanmin(covar_space)) / (np.nanmax(covar_space) - np.nanmin(covar_space))

    covdistr = CovarDistribution(model, loc=loc_expr, scale=scale_expr, c=c)

    x = pd.DataFrame({"t": covar_time_norm, "s": covar_space_norm})

    output = covdistr(x)

    return output


#----- For Spatially and Temporally NS-GEV model with constant sigma/mu ratio fitted all at once (USELESS) -----#

def get_vdistr_disp_norm_st_all(model, params, covar_time, covar_space):
    """generate a variable distribution with time- and space-dependent parameters for constant sigma/mu models"""
    assert isinstance(params, dict), 'parameters must be stored in dict'

    loc = params['loc']
    scale = params['scale']
    c = params['c']

    assert len(loc) == 3 or len(scale) == 3, 'at least one of the parameters must be [time_slope, space_slope, intercept] shaped for ST-NS-GEV model fit'
    # -> add the case where one parameter is not space-dependent

    if len(loc) == 2:       # space-dependent only location parameter
        loc_s_sl = loc[0]
        loc_s_int = loc[1]
        loc_expr = MathExpr('{0}*s + {1}'.format(loc_s_sl, loc_s_int))

    elif len(loc) == 3:     # space- and time-dependent location parameter
        loc_t_sl = loc[0]
        loc_s_sl = loc[1]
        loc_int = loc[2]
        loc_expr = Linear_Combination({'intercept': loc_int, 't': loc_t_sl, 's': loc_s_sl})

    scale_s_sl = scale[0]
    scale_int = scale[1]

    scale_expr = MathExpr('{0} + {1}*s + {2}*({0}+{1}*s) / ({3}+{4}*s) * t'.format(scale_int, scale_s_sl, loc_t_sl, loc_int, loc_s_sl))

    # normalize covariates
    covar_time_norm = (covar_time - np.nanmin(covar_time)) / (np.nanmax(covar_time) - np.nanmin(covar_time)) - 0.5
    covar_space_norm = (covar_space - np.nanmin(covar_space)) / (np.nanmax(covar_space) - np.nanmin(covar_space))

    covdistr = CovarDistribution(model, loc=loc_expr, scale=scale_expr, c=c)

    x = pd.DataFrame({"t": covar_time_norm, "s": covar_space_norm})

    output = covdistr(x)

    return output


#--------------------------------- For full Spatially and Temporally NS-GEV models ------------------------------#

def get_vdistr_norm_st_nsgev_full(model, params, covar_time, covars_space):
    """generate a variable distribution with time- and space-dependent parameters"""
    assert isinstance(params, dict), 'parameters must be stored in dict'

    loc = params['loc']
    scale = params['scale']
    c = params['c']

    covar_lat = covars_space['lats'].values
    covar_lon = covars_space['lons'].values

    assert len(loc) == 4 or len(scale) == 4, 'at least one of the parameters must be [time_slope, lat_slope, lon_slope, intercept] shaped for full ST-NS-GEV model fit'
    # -> add the case where one parameter is not space-dependent

    if len(loc) == 3:  # loc(lat, lon)
        loc_lat_sl = loc[0]
        loc_lon_sl = loc[1]
        loc_int = loc[2]
        loc_expr = Linear_Combination({'intercept': loc_int, 'lat': loc_lat_sl, 'lon': loc_lon_sl})

    elif len(loc) == 4:  # loc(t, lat, lon)
        loc_t_sl = loc[0]
        loc_lat_sl = loc[1]
        loc_lon_sl = loc[2]
        loc_int = loc[3]
        #loc_expr = Linear_Combination({'intercept': loc_int, 't': loc_t_sl, 'lat': loc_lat_sl, 'lon': loc_lon_sl})
        # FOR TEST: mu(s,t) = mu(s)(1+mu3*t)
        loc_expr = MathExpr('({0} + {1}*lat + {2}*lon) * (1 + {3}*t)'.format(loc_int, loc_lat_sl, loc_lon_sl, loc_t_sl))

    if len(scale) == 3:  # scale(lat, lon)
        scale_lat_sl = scale[0]
        scale_lon_sl = scale[1]
        scale_int = scale[2]
        scale_expr = Linear_Combination({'intercept': scale_int, 'lat': scale_lat_sl, 'lon': scale_lon_sl})

    elif len(scale) == 4:  # scale(t, lat, lon)
        scale_t_sl = scale[0]
        scale_lat_sl = scale[1]
        scale_lon_sl = scale[2]
        scale_int = scale[3]
        #scale_expr = Linear_Combination({'intercept': scale_int, 't': scale_t_sl, 'lat': scale_lat_sl, 'lon': scale_lon_sl})
        # FOR TEST: scale(s,t) = scale(s)(1+scale3*t)
        scale_expr = MathExpr('({0} + {1}*lat + {2}*lon) * (1 + {3}*t)'.format(scale_int, scale_lat_sl, scale_lon_sl, scale_t_sl))

    # normalize covariates
    covar_time_norm = (covar_time - np.nanmin(covar_time)) / (np.nanmax(covar_time) - np.nanmin(covar_time)) - 0.5
    covar_lat_norm = (covar_lat - np.nanmin(covar_lat)) / (np.nanmax(covar_lat) - np.nanmin(covar_lat))
    covar_lon_norm = (covar_lon - np.nanmin(covar_lon)) / (np.nanmax(covar_lon) - np.nanmin(covar_lon))

    covdistr = CovarDistribution(model, loc=loc_expr, scale=scale_expr, c=c)

    x = pd.DataFrame({"t": covar_time_norm, "lat": covar_lat_norm, "lon": covar_lon_norm})

    output = covdistr(x)

    return output


#------------------ For full Spatially and Temporally NS-GEV model with constant sigma/mu ratio  --------------------#

def get_vdistr_disp_norm_st_full(model, params, covar_time, covars_space):
    """generate a variable distribution with time- and space-dependent parameters"""
    assert isinstance(params, dict), 'parameters must be stored in dict'

    loc = params['loc']
    scale = params['scale']
    c = params['c']

    covar_lat = covars_space['lats'].values
    covar_lon = covars_space['lons'].values

    assert len(loc) == 4 or len(scale) == 4, 'at least one of the parameters must be [time_slope, lat_slope, lon_slope, intercept] shaped for full ST-NS-GEV model fit'
    # -> add the case where one parameter is not space-dependent

    if len(loc) == 3:  # loc(lat, lon)
        loc_lat_sl = loc[0]
        loc_lon_sl = loc[1]
        loc_int = loc[2]
        loc_expr = Linear_Combination({'intercept': loc_int, 'lat': loc_lat_sl, 'lon': loc_lon_sl})

    elif len(loc) == 4:  # loc(t, lat, lon)
        loc_t_sl = loc[0]
        loc_lat_sl = loc[1]
        loc_lon_sl = loc[2]
        loc_int = loc[3]
        #loc_expr = Linear_Combination({'intercept': loc_int, 't': loc_t_sl, 'lat': loc_lat_sl, 'lon': loc_lon_sl})
        # FOR TEST: mu(s,t) = mu(s)(1+mu3*t)
        loc_expr = MathExpr('({0} + {1}*lat + {2}*lon) * (1 + {3}*t)'.format(loc_int, loc_lat_sl, loc_lon_sl, loc_t_sl))

    scale_lat_sl = scale[0]
    scale_lon_sl = scale[1]
    scale_int = scale[2]

    #scale_expr = MathExpr('{0} + {1}*lat + {2}*lon + {3}*({0} + {1}*lat + {2}*lon) / ({4} + {5}*lat + {6}*lon) * t'.format(scale_int, scale_lat_sl, scale_lon_sl, loc_t_sl, loc_int, loc_lat_sl, loc_lon_sl))
    # FOR TEST: scale(s,t) = scale(s)(1+scale3*t)
    scale_expr = MathExpr('({0} + {1}*lat + {2}*lon) * (1 + {3}*t)'.format(scale_int, scale_lat_sl, scale_lon_sl, loc_t_sl))


    # normalize covariates
    covar_time_norm = (covar_time - np.nanmin(covar_time)) / (np.nanmax(covar_time) - np.nanmin(covar_time)) - 0.5
    covar_lat_norm = (covar_lat - np.nanmin(covar_lat)) / (np.nanmax(covar_lat) - np.nanmin(covar_lat))
    covar_lon_norm = (covar_lon - np.nanmin(covar_lon)) / (np.nanmax(covar_lon) - np.nanmin(covar_lon))

    covdistr = CovarDistribution(model, loc=loc_expr, scale=scale_expr, c=c)

    x = pd.DataFrame({"t": covar_time_norm, "lat": covar_lat_norm, "lon": covar_lon_norm})

    output = covdistr(x)

    return output


#----------------------------------------------------------------------------------------------------------#


    ##############################
    #                            #
    #     Original co-variate    #
    #                            #
    ##############################


def get_vdistr(model, params, years):
    """generate a variable distribution"""
    assert isinstance(params, dict), 'parameters must be stored in dict'
    loc = params['loc']
    scale = params['scale']
    c = params['c']

    assert len(loc) == 2 or len(scale) == 2, 'at least one of the parameters must be [slope, intercept] shaped for NS distribution'

    if len(loc) == 2:
        loc_sl = loc[0]
        loc_int = loc[1]
        loc_expr = LinearExpr([loc_sl, loc_int])
    else:
        loc_expr = loc[0]

    if len(scale) == 2:
        scale_sl = scale[0]
        scale_int = scale[1]
        scale_expr = LinearExpr([scale_sl, scale_int])
    else:
        scale_expr = scale[0]

    vdistr = VarDistribution(model, loc=loc_expr, scale=scale_expr, c=c)
    x_ = np.arange(0, len(years))
    x = pd.DataFrame({"x":x_})

    output = vdistr(x)

    return output

