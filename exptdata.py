#!/usr/bin/env python

# Centralised code to set model and experiment names.

# This is just an initial stab at it. More functionality is planned:
# https://github.com/COSIMA/ACCESS-OM2-1-025-010deg-report/issues/6

# load this with
#     import exptdata

import cosima_cookbook as cc

from collections import OrderedDict
import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
import yaml
import f90nml  # from https://f90nml.readthedocs.io/en/latest/



# define common start and end year for climatologies (Jan 1 assumed)
clim_tend = pd.to_datetime('2019', format='%Y')
clim_tstart = clim_tend - pd.DateOffset(years=20)

# as in Kiss et al (GMD, 2020)
clim_tend = pd.to_datetime('2018', format='%Y')
clim_tstart = clim_tend - pd.DateOffset(years=25)  # 1993



basedir = '/g/data/hh5/tmp/cosima/'

# Model data sources.
# More experiments (or variables each experiment) can be added here if needed.
# locals().update(exptdata.exptdict['1deg']) will define all variables for the '1deg' experiment (dangerous!).
# exptdir = exptdata.expdict[expkey]['exptdir'] etc is safer.
# desc is a short descriptor for use in figure titles.
# n_files is a negative number - designed to find data from just the last IAF cycle.
# Uses OrderedDict so that iteration on exptdict will be in this order.
# NB: offset changes effect of time_units: https://github.com/COSIMA/cosima-cookbook/issues/113
# Also MOM and CICE have different time_units: https://github.com/COSIMA/access-om2/issues/117#issuecomment-446465761
# so the time_units specified here may need to be overridden when dealing with CICE data - e.g. see ice_validation.ipynb

exptdict = OrderedDict([
    ('1deg',   {'model': 'access-om2',
                'expt': '1deg_jra55_iaf_omip2_cycle', # cycle number appended below
                'exptdir': '/g/data/ik11/outputs/access-om2/1deg_jra55_iaf_omip2_cycle', # cycle number appended below
#                 'expt': '1deg_jra55_iaf_omip2-fixed_cycle', # cycle number appended below
#                 'exptdir': '/scratch/v45/aek156/access-om2/archive/1deg_jra55_iaf_omip2-fixed_cycle', # lots of symlinks to /scratch/v45/hh0162/access-om2/archive/1deg_jra55_iaf_omip2-fixed ; cycle number appended below
#                 'dbpath': '/g/data/v45/aek156/notebooks/github/aekiss/ice_analysis/1deg_jra55_iaf_omip2-fixed.db',
                'desc': 'ACCESS-OM2',
                'n_files': -12,
                'time_units': 'days since 1718-01-01',
                'offset': -87658,
                'gridpaths': ['/g/data/ik11/grids/ocean_grid_10.nc'],
                'cycles': 6
               }),
    ('025deg', {'model': 'access-om2-025',
                'expt': '025deg_jra55_iaf_omip2_cycle', # cycle number appended below
                'exptdir': '/g/data/ik11/outputs/access-om2-025/025deg_jra55_iaf_omip2_cycle', # cycle number appended below
#                 'expt': '025deg_jra55_iaf_amoctopo_cycle', # cycle number appended below
#                 'exptdir': '/scratch/e14/rmh561/access-om2/archive/025deg_jra55_iaf_amoctopo_cycle', # cycle number appended below
#                 'dbpath': '/scratch/e14/rmh561/access-om2/archive/databases/cc_database_omip',
                'desc': 'ACCESS-OM2-025',
                'n_files' : -34,
                'time_units': 'days since 1718-01-01',
                'offset': -87658,
                'gridpaths': ['/g/data/ik11/grids/ocean_grid_025.nc'],
                'cycles': 6
               }),
    ('01deg',  {'model': 'access-om2-01',
                'expt': '01deg_jra55v140_iaf',
                'exptdir': '/g/data/cj50/access-om2/raw-output/access-om2-01/01deg_jra55v140_iaf',
#                 'dbpath': '/g/data/ik11/databases/cosima_master.db',
                'desc': 'ACCESS-OM2-01',
                'n_files': None,
                'time_units': 'days since 0001-01-01',
                'offset': None,
                'gridpaths': ['/g/data/ik11/grids/ocean_grid_01.nc', '/g/data/cj50/access-om2/raw-output/access-om2-01/01deg_jra55v140_iaf/output000/ocean/ocean-2d-area_t.nc'],
                'cycles': 3
               }),
# These are the older version 1 results, as used in Kiss et al 2020
    ('1degv1',   {'model': 'access-om2',
                'expt': '1deg_jra55v13_iaf_spinup1_B1',
                'desc': 'ACCESS-OM2 v1',
                'n_files': -12,
                'time_units': 'days since 1718-01-01',
                'offset': -87658,
                'gridpaths': ['/g/data/ik11/grids/ocean_grid_10.nc']}),
    ('025degv1', {'model': 'access-om2-025',
                'expt': '025deg_jra55v13_iaf_gmredi6',
                'desc': 'ACCESS-OM2-025 v1',
                'n_files': -34,
                'time_units': 'days since 1718-01-01',
                'offset': -87658,
                'gridpaths': ['/g/data/ik11/grids/ocean_grid_025.nc']}),
    ('01degv1',  {'model':'access-om2-01',
                'expt': '01deg_jra55v13_iaf',
                'desc': 'ACCESS-OM2-01 v1',
                'n_files': None,
                'time_units': 'days since 0001-01-01',
                'offset': None,
                'gridpaths': ['/g/data/ik11/grids/ocean_grid_01.nc', '/g/data/cj50/access-om2/raw-output/access-om2-01/01deg_jra55v140_iaf/output000/ocean/ocean-2d-area_t.nc']})
])


# add parameter ensembles
for res in ['1deg']:
    ens_dirs = glob.glob('/scratch/jk72/aek156/access-om2/archive/'+res+'*')
    ens_dirs.sort()
    template = os.path.basename(os.path.commonprefix(ens_dirs))
    prefix = template.split('_')[0]
    for d in ens_dirs:
        f = os.path.basename(d)

        if f == template:
            ekey = '_'.join([prefix, 'control'])
        else:
            ekey = ''.join([prefix, f[len(template):]])

        exptdict[ekey] = {
            'expt': f,
            'dbpath': '/home/156/aek156/payu/param_ensemble/param_ensemble.db',
            'desc': '='.join(ekey.replace('_', ' ').rsplit(' ', 1)),
            }

        if res == '1deg':
            exptdict[ekey].update({
                'model': 'access-om2',
                'exptdir': '/scratch/jk72/aek156/access-om2/archive/' + f,
                'gridpaths': ['/g/data/ik11/grids/ocean_grid_10.nc'],
                'ensemble': '/home/156/aek156/payu/param_ensemble/access-om2/ensemble/ensemble.yaml'
                })
        elif res == '025deg':
            exptdict[ekey].update({
                'model': 'access-om2-025',
                'exptdir': '/scratch/jk72/aek156/access-om2-025/archive/' + f,
                'gridpaths': ['/g/data/ik11/grids/ocean_grid_025.nc'],
                'ensemble': '/home/156/aek156/payu/param_ensemble/access-om2-025/ensemble/ensemble.yaml'
                })
# TODO: support for 0.1deg

        if f == template:
            exptdict[ekey]['desc'] = exptdict[ekey]['desc'].replace('=', ' ')
            exptdict[ekey]['perturbed'] = None
        else:
            exptdict[ekey]['perturbed'] = '_'.join(f[len(template)+1:].split('_')[:-1])  # the variable that was perturbed (also a key into params)

        outputpaths = glob.glob(os.path.join(exptdict[ekey]['exptdir'], 'output*'))
        outputpaths.sort()

        if len(outputpaths) <= 1:
            # no output, so ignore this experiment
            del exptdict[ekey]
            continue


# For each experiment, record the values of all parameters that are perturbed in any ensemble member
        exptdict[ekey]['params'] = dict()
        ensemble = yaml.load(open(exptdict[ekey]['ensemble'], 'r'), Loader=yaml.SafeLoader)
        for fname, nmls in ensemble['namelists'].items():
            fpath = os.path.join(outputpaths[-1], fname)
            try:
                nml = f90nml.read(fpath)
            except FileNotFoundError:
                continue
            templatenml = f90nml.read(os.path.join(os.path.dirname(exptdict[ekey]['ensemble']), ensemble['template'], fname))
            for group, names in nmls.items():
                for name in names:
                    turningangle = set([fname, group, name]) == set(['ice/cice_in.nml', 'dynamics_nml', 'turning_angle'])
                    if turningangle:
                        exptdict[ekey]['params']['turning_angle'] = np.arcsin(nml[group]['sinw']) * 180. / np.pi
                    else:
                        try:
                            exptdict[ekey]['params'][name] = nml[group][name]
                        except KeyError:  # fill in from template - ASSUMES TEMPLATE VALUE IS WHAT WAS USED!
                            exptdict[ekey]['params'][name] = templatenml[group][name]


# Add sessions where they don't already exist.
# TODO: reuse sessions from previous experiments if dbpath is the same
for e in exptdict.values():
    if not('dbpath' in e):
        e['session'] = cc.database.create_session()
for e in exptdict.values():
    if not('session' in e):
        e['session'] = cc.database.create_session(e['dbpath'])

# Add expdirs where they don't already exist.
for e in exptdict.values():
    if not('exptdir' in e):
        e['exptdir'] = os.path.join(basedir, e['model'], e['expt' ])


# Add cycles where they don't already exist.
for e in exptdict.values():
    if not('cycles' in e):
        e['cycles'] = 1


# set up multiple cycles
for k, e in exptdict.items():
    exptdict[k] = [e.copy() for c in range(e['cycles'])]

for ekey in ['1deg', '025deg']:
    for c, ec in enumerate(exptdict[ekey]):
        ec['expt'] += str(c+1)
        ec['exptdir'] += str(c+1)

exptdict['01deg'][1]['expt'] += '_cycle2'
exptdict['01deg'][1]['exptdir'] += '_cycle2'

exptdict['01deg'][2]['expt'] = '01deg_jra55v140_iaf_cycle3'
exptdict['01deg'][2]['exptdir'] = '/scratch/v45/aek156/access-om2/archive/01deg_jra55v140_iaf_cycle3'
exptdict['01deg'][2]['dbpath'] = '/g/data/v45/aek156/notebooks/github/aekiss/CC_sandbox/cyc3_database_analysis3-20p07.db'
exptdict['01deg'][2]['session'] = cc.database.create_session(exptdict['01deg'][2]['dbpath'])



#################################################################################################
# functions to share across all notebooks

def joinseams(d, lon=False, tripole_flip=False):
    """
    Concat duplicated western edge data along eastern edge and flipped data along tripole seam
    to avoid gaps in plots.
    Assumes the last dimension of d is x and second-last is y.

    d: xarray.DataArray or numpy.MaskedArray (or numpy.Array - UNTESTED!)

    lon: boolean indicating whether this is longitude data in degrees
        (in which case 360 is added to duplicated eastern edge).
        Ignored if d is a DataArray.

    tripole_flip: boolean indicating whether to reverse duplicated data on tripole seam.
        You'd normally only do this with coord data.
        Ignored if d is a DataArray.

    Returned array shape has final 2 dimensions increased by 1.
    """
    if type(d) is xr.core.dataarray.DataArray:
        dims = d.dims
        out = xr.concat([d, d.isel({dims[-1]: 0})], dim=dims[-1])
        out = xr.concat([out, out.isel({dims[-2]: -1})], dim=dims[-2])
    elif type(d) is np.ma.core.MaskedArray:
        dims = range(len(d.shape))
        if lon:
            out = np.ma.concatenate([d, d[:,0,None]+360], axis=-1)
        else:
            out = np.ma.concatenate([d, d[:,0,None]], axis=-1)
        if tripole_flip:
            out = np.ma.concatenate([out, np.flip(out[None,-1,:], axis=-1)], axis=-2)
        else:
            out = np.ma.concatenate([out, out[None,-1,:]], axis=-2)
    else:  # NB: UNTESTED!!
        assert type(d) is np.ndarray
        dims = range(len(d.shape))
        if lon:
            out = np.concatenate([d, d[:,0,None]+360], axis=-1)
        else:
            out = np.concatenate([d, d[:,0,None]], axis=-1)
        if tripole_flip:
            out = np.concatenate([out, np.flip(out[None,-1,:], axis=-1)], axis=-2)
        else:
            out = np.concatenate([out, out[None,-1,:]], axis=-2)
    return out

#################################################################################################


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=
        'Centralised definition of model and experiment names.')
    parser.add_argument('-l', '--latex',
                        action='store_true', default=False,
                        help='Output data as latex table')
    if vars(parser.parse_args())['latex']:
        print(r'''
\begin{tabularx}{\linewidth}{lXp{0.4\linewidth}}
\hline
\textbf{Configuration} & \textbf{Experiment} & \textbf{Path to output data on NCI} \\
\hline
''')
        for k in exptdict.keys():
            e = exptdict[k]
            print(r'{} & {} & {}\\'.format(e['desc'].replace('°','$^\circ$'), e['expt'],
                      r'\texttt{' + e['exptdir'].replace('/', '\\slash ') + r'}'))
        print(r'''
\hline
\hline
\end{tabularx}''')
    else:
        print(' '.join(e for e in exptdirs), end='')  # for use in get_namelists.sh
