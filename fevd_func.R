### INSTALL R PACKAGES WITH RPY2  ###
# from rpy2.robjects.packages import importr
# utils = importr('utils')
# utils.install_packages('nomdupackage')


#library(RJSONIO)
library(extRemes)

# USELESS
#to_json <- function(x){
#    out = toJSON(x) 
#    return(out)
#}


make_fevd_s <- function(x, method){
    out = fevd(x, method=method)
    return(out)
}


make_fevd_ns <- function(x, data, loc_fun, sca_fun, method){
    out = fevd(x, data, threshold=NULL, location.fun=loc_fun, scale.fun=sca_fun, type=c("GEV"), method=method)
    return(out)
}
#method=c("MLE")

# USELESS
get_qcov <- function(nsgev, nrows){
    out = make.qcov(nsgev, nr=nrows)
    return(out)
}


get_pars_with_ci <- function(nsgev){
    out = ci(nsgev, alpha=0.1, type=c("parameter"))
    return(out)
}


get_return_levels_norm <- function(nsgev, ts, qcov){
    out = return.level(nsgev, return.period=ts, do.ci=TRUE, qcov=qcov, method=c('normal'), alpha=0.1)
    return(out)
}
# method=c("boot")  c("normal")


get_return_levels_boot <- function(nsgev, ts){
    out = return.level(nsgev, return.period=ts, do.ci=TRUE, method=c('boot'), alpha=0.1, return.samples=TRUE)
    return(out)
}


get_ci_norm <- function(nsgev, t, qcov){
    out = ci(nsgev, alpha=0.1, qcov=qcov, type=c('return.level'), return.period=t, method="normal")
    return(out)
}


get_ci_boot <- function(nsgev, t){
    out = ci(nsgev, alpha=0.1, type=c('return.level'), return.period=t, method="boot", return.samples=TRUE)
    return(out)
}


get_lh_prof <- function(fevd){
    out = profliker(fevd, type='parameter', which.par=3)
    return(out)
}
