"""Data treatment config file"""

import os

DIRNAME = os.path.dirname(__file__)

OUTDIR = os.path.join(DIRNAME, "output")
DATADIR = os.path.join(OUTDIR, "data")
FIGDIR = os.path.join(OUTDIR, "figures")


### CST ###

#durations = ['1h', '2h', '3h', '4h', '6h', '8h', '10h', '12h', '15h', '18h', '24h']
#am_res = [0.25, 0.5, 1, 2]  # valid spatial resolution for am series calculation
