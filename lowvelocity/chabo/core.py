# -*- coding: utf-8 -*-
"""
Created on Wed Sep 02 11:26:02 2015

@author: ISTI_EW
test fmm
"""

from __future__ import unicode_literals, division, absolute_import, print_function
from six import string_types
import skfmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
import chabo.velocitygrid as vg
import chabo.inputs as inp
import sys
import tables
import os
import scipy
import scipy.ndimage
from collections import namedtuple

from scipy.ndimage.interpolation import zoom
from scipy.linalg import norm

import pudb

import warnings


# warnings.filterwarnings('error')
class Chabo(object):
    """
    Main container class for Chabo. Defines geometry, sets velocities,
    and embeds stations/events\n
    Parameters
    ----------
    xmin : list-like (list, tuple, np.ndarray)
        The minimum geometric coordinates for the grid space. Either local
        coordinates (in meters) or lat lons can be used.
        xmin can either be length 2 or 3 (for 2d or 3d case) and must be
        the same length as xmax, dx, xnum
    xmax : list-like (list, tuple, np.ndarray)
        Same requirements as xmin but represents upper extents of geometry
    dx : list-like (list, tuple, np.ndarray)
        The grid spacing in meters for either 2 or 3 dimensions.
        For example, [20,15] would imply 20m in the x direction and 15m in
        the y direction.
    xnum : list-like (list, tuple, np.ndarray)
        The number of blocks to use in each direction
    latlon : bool
        If true values in lims are lon, lat, elevation so they will be
        converted to local coordinates first.
    VelMod : input for chabo.velocityGrid to make velocity model
        If a velocity model is not passed one will be crreated
    vtype : str or None
        The type of velocity model. Options are: '1d', '1dcsv', '2dgrid',
        '2dcsv', '3dgrid', '3dcsv', '3dnll'. If None chabo will try to
        determine the type automagically.
    stations : pandas dataframe
        Data frame with the following fields: 'ID','X','Y','Z', each row
        corresponding to a different station
    events : pandas dataframe
        Same as stations but for events
    phases : str, pandas dataframe, or none
        Path to the phase file or pandas dataframe containing phase info
    phase_type : str
        the phase type of the velocity model (P or S)
    squash : bool
        If True convert all station and event coordinates to 2d if the chabo
        instance is 2d.
    hdf5_path : str
        The path to an hdf5 store of travel time grids

    Notes
    ------
    Of the 4 geometric input parameters [xmin, xmax, dx, xnum] exactly
    3 must be list like objects and of the same length, either 2 for 2
    dimensions or 3 for 3 dimensions.

    Acceptable combinations are:
    xmin,xmax,xnum
    xmin, xmax, dx
    xmin,dx,xnum

    """
    ray_columns = ['station', 'event', 'turningPoints', 'startTime']
    spakey = ['X', 'Y', 'Z']
    apkey = ['Xap', 'Yap', 'Zap']
    inkey = ['Xin', 'Yin', 'Zin']


    def __init__(self, xmin=None, xmax=None, dx=None, xnum=None, latlon=False,
                 velMod=None, vtype=None, stations=None, events=None,
                 phases=None, phase_type='P', squash=True, hdf5_path='.chabo.h5',
                 **kwargs):
        # convert to local coords (cartesian) if required
        self.latlon = latlon
        if self.latlon:
            xmin, xmax = self._convert2local(xmin, xmax)
        # set geometric properties
        self.xmin = xmin
        mod_params = self._mesh_grid(xmin, xmax, dx, xnum)
        self.X, self.xls, self.xmax, self.dx, self.xnum = mod_params
        # instantiate some variables, trunicate key lists to actual dimensions
        self.phase_type = phase_type
        self.spakey = self.spakey[:len(self.X)]  # columns for spatial dims
        self.apkey = self.apkey[:len(self.X)]  # columns for approx. locations
        self.inkey = self.inkey[:len(self.X)]  # columns for indicies
        self._rms = None
        # init container for travel time grids (keys are event IDs)
        self.tts = {}
        self.nested_tts = {}  # used for velocity iterations
        self.station_tts = {}
        self.hdf5_path = hdf5_path
        # init dataframe to contain ray paths for each event-station pair
        self.rays = pd.DataFrame(columns=self.ray_columns)
        ## Attach required files if they were given, else set as 0
        self.stations = self.get_stations(stations, **kwargs)
        self.events = self.get_events(events, **kwargs)
        self.phases = self.get_phases(phases, **kwargs)
        self.velMod = self.get_velocity_model(velMod, vtype=vtype, **kwargs)

    @property
    def rms(self):
        self.calc_residuals()
        return self._rms

    ### Geometry creation and input checks

    def _check_inputs(self, xmin, xmax, dx, xnum):
        # make sure xmin is not greater than xmax if defined
        if all([isinstance(x, (list, tuple, np.ndarray)) for x in [xmin, xmax]]):
            if any([x[0] > x[1] for x in zip(xmin, xmax)]):
                raise ValueError('xmin cannot be greater than xmax')
        checkList = np.array([isinstance(x, (list, tuple, np.ndarray))
                              for x in [xmin, xmax, dx, xnum]])
        if checkList.sum() != 3:  # make sure exactly 3 inputs are defined
            msg = ('Exactly 3 of the following variables must be defined, the '
                   'fourth left as none: xmin, xmax, dx, xnum')
            raise ValueError(msg)
        if len(set([len(x) for x in [xmin, xmax, dx, xnum] if isinstance(x,
                                                                         (list, tuple,
                                                                          np.ndarray))])) > 1:  # all inputs equal-lengthed
            msg = 'All 3 input parameters used must be the same length'
            raise ValueError(msg)
        # acceptable binary combos of input parameters
        acceptableCombos = np.array([[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1]])
        if not 3 in np.dot(acceptableCombos, checkList):
            msg = ('Unsupported combination of input parameters selected, see '
                   'notes on Chabo class for excepted combos')
            raise Exception(msg)
        return checkList

    def _mesh_grid(self, xmin, xmax, dx, xnum):  # Mesh grids
        checkList = self._check_inputs(xmin, xmax, dx, xnum)
        # if xmin, xmax and xnum defined
        if all(checkList == [1, 1, 0, 1]):
            lims = zip(xmin, xmax, xnum)  # zip parameters together
            xls = [np.linspace(x[0], x[1], num=x[2]) for x in lims]
            dx = [(float(x[1]) - x[0]) / x[2] for x in lims]
        # if xmin, xmax, dx defined
        if all(checkList == [1, 1, 1, 0]):
            lims = zip(xmin, xmax, dx)
            xls = [np.arange(x[0], x[1] + x[2], x[2]) for x in lims]
            # Make sure no points beyond end of grid are included
            xls = [x[:-1] if x[-1] > xmax[num] else x for num, x in enumerate(xls)]
            xnum = [len(x) for x in xls]
        # if  xmin, dx and xnum defined
        if all(checkList == [1, 0, 1, 1]):
            lims = zip(xmin, dx, xnum)
            xmax = [x[0] + x[1] * (x[2] - 1) for x in lims]
            lims = zip(xmin, xmax, xnum)
            xls = [np.linspace(x[0], x[1], num=x[2]) for x in lims]
        X = np.meshgrid(*xls, indexing='ij')
        return X, xls, xmax, dx, xnum

    def _convert2local(self, lims):
        # convert lat lons to local coordinates
        raise Exception('Not yet implimented')  # TODO impliment

    ###### Attach functions (Vmodel, phases, stations, events, etc.)

    def get_velocity_model(self, velmod, vtype=None, **kwargs):
        if velmod is None:
            return None
        return vg.VelocityGrid(velmod, self, vtype=vtype, **kwargs)

    def attach_velocity_model(self, velmod, vtype=None, **kwargs):
        self.velMod = self.get_velocity_model(velmod, vtype=vtype, **kwargs)

    def get_stations(self, sfile, stype='df', **kwargs):
        if sfile is None:  # if None set as None
            return None
        if isinstance(sfile, string_types):  # if str read csv
            sfile = pd.read_csv(sfile)
        stadf = inp.Stations(sfile, self, stype=stype, **kwargs)
        if self.latlon:
            stadf = self._converToLocal(stadf)
        stadf.index = stadf.ID
        # make sure locations are within grid
        for num, row in stadf.iterrows():
            self._check_bounds(row)
        return self._get_approx_locations(stadf)

    def attach_stations(self, sfile, stype='df', **kwargs):
        self.stations = self.get_stations(sfile, stype=stype, **kwargs)

    def get_events(self, efile, etype='df', **kwargs):
        if efile is None:
            return None
        if isinstance(efile, string_types):
            efile = pd.read_csv(efile)
        evdf = inp.Events(efile, self, etype=etype, **kwargs)
        if self.latlon:
            evdf = self._converToLocal(evdf)
        evdf.index = evdf.ID
        # make sure locations are within grid
        for num, row in evdf.iterrows():
            self._check_bounds(row)
        return self._get_approx_locations(evdf)

    def attach_events(self, efile, etype='df', **kwargs):
        self.events = self.get_events(efile, etype=etype, **kwargs)

    def get_phases(self, phases, ptype='df', **kwargs):
        if phases is None:
            return None
        if isinstance(phases, string_types):
            phases = pd.read_csv(phases)
        # attach phases and normalize TT colulmn by subtracting origin time
        phases = inp.Phases(phases, self, ptype=ptype, **kwargs)
        # temp event df
        if (abs(phases.TT)).median() > 6 * 60:  # determine if origins are ts
            et = self.events[['O']]
            et['EID'] = self.events.ID
            pt = phases.merge(et, on='EID')
            pt['TT'] = pt['TT'] - pt['O']
            phases = pt.drop('O', axis=1)
        else:
            phases = phases
        return phases

    def attach_phases(self, phases, ptype='df', **kwargs):
        self.phases = self.get_phases(phases, ptype=ptype, **kwargs)

    # Check if any station or event locations are outside of grid
    def _check_bounds(self, row, dimnames=['X', 'Y', 'Z']):
        appdims = np.array([row[dimnames[dinum]] for dinum in
                            range(len(self.xmin))])
        if any((appdims > self.xmax) | (appdims < self.xmin)):
            msg = (('The following row of the events or stations dataframe '
                    'does not fit in the dimension limits of %s to %s: '
                    '\n%s \n ') % (self.xmin, self.xmax, row))
            raise ValueError(msg)

    ### Embed stations/events in grid

    def _get_approx_locations(self, df, grid=None):
        """
        loop through the dataframe (either station or events) and find
        nearest location in grid, append approximate locations and
        indicies. Can pass a custom grid
        """
        if grid is None:
            grid = self
        for num, row in df.iterrows():
            coords = [row[x] for x in self.spakey]
            xind = [abs(coords[num2] - grid.xls[num2]).argmin() for num2 in
                    range(len(grid.xls))]  # index of best fitting x point in L1
            for num2 in range(len(grid.xls)):
                df.loc[num, self.apkey[num2]] = grid.xls[num2][xind[num2]]
                df.loc[num, self.inkey[num2]] = int(xind[num2])
            df.loc[num, 'embed_error'] = self._get_embed_errors(df.loc[num])
        # cast dtypes
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except Exception:
                pass
        for in_col in self.inkey:
            df[in_col] = df[in_col].astype(np.int)
        return df.sort_index()

    def _get_embed_errors(self, row):
        """
        Get the error associated with embedding the station/event in a grid
        (in grid distance units)
        """
        return norm([row[x] - row[y] for x, y in zip(self.spakey, self.apkey)])

    def id2event(self, eid):
        """
        Given an event id (eid) return the row of that event
        """

        try:
            return self.events.loc[eid]
        except IndexError:
            msg = 'No events with ID %s found' % eid
            raise IndexError(msg)

    def ind2event(self, ind):
        return self.events.iloc[ind]

    def id2station(self, sid):
        """
        Given an station id (sid) return the row of that station
        """
        try:
            return self.stations.loc[sid]
        except IndexError:
            msg = 'No events with ID %s found' % sid
            raise IndexError(msg)

    def ind2station(self, ind):
        return self.stations.iloc[ind]

    ### Fast marching and ray tracing functions

    def fast_march(self, get_rays=True, num_blocks=10):
        """
        Call the fast marching method calculate travel times in each grid point
        for a given event location
        Parameters
        ----------
        get_rays : bool
            If true attempt to back calculate the ray paths based on
            travel time grid
        num_blocks : int
            The number of neighbors to include for determining path
            tracing back to origin
        """
        # make sure that stations and events are defined
        con1 = (self.events is None or not len(self.events))
        con2 = (self.stations is None or not len(self.stations))
        if con1 or con2:
            msg = ('No events or no stations passed, check events and stations'
                   'dataframes')
            raise ValueError(msg)
        # make sure a valid velocity model is found
        if self.velMod is None:
            msg = 'Undefined velocity model, define it with attach_velocity_model'
            raise ValueError(msg)
        # If phases are attached
        if self.phases is not None:
            # get events for phase picks exist
            eventdf = self.events[self.events.ID.isin(set(self.phases.EID))]
        else:  # Else use all events
            eventdf = self.events
        for evnum, evrow in eventdf.iterrows():  # Loop through each event
            phi = self._get_phi(evrow)
            tt = skfmm.travel_time(phi, self.velMod, dx=self.dx)
            self.tts[evrow.ID] = tt
            if get_rays:  # If ray paths are to be back calculated
                # if phases are attached
                if self.phases is not None:
                    # get phases on curent event
                    evephases = self.phases[self.phases.EID == evrow.ID]
                    # Stations which have phases for current event
                    stadf = self.stations[self.stations.ID.isin(evephases.SID)]
                else:  # If no phases use all station-event pairs
                    stadf = self.stations

                for stanum, starow in stadf.iterrows():  # loop through each station
                    sr = self._get_ray_paths(starow, tt, evrow, num_blocks)
                    self.rays.loc[len(self.rays)] = sr

    def iter_models(self, vmods, parallel=True, cores=-1, output='residual'):
        """
        Iterate through a series of vmodes, return travel times at stations
        or residuals. Note; A new velocity grid is defined for each station
        in order to minimize the number of points that have to be calculated,
        The number of grid points, however, remains constaint

        parameters
        ----------
        vmods : iterator
            an iterator of velocity model grids and modnumber tuple
        parallel : bool
            If True run all models in parallel using job lib's multiprocessing
        cores : int
            The number of cores to use in the calculatio
        output : str
            The output type, either "residuals" or "travel_times"
        Returns
        --------
        Results are returned as a flat dataframe with the following row:
        MID, SID, EID, travel_time, residual
        """
        # put import here so that joblib is not required
        from joblib import Parallel, delayed
        # Init key vars
        dx = self.dx
        phi = self._get_phi(self.stations.iloc[0])
        ein = [tuple(self.events[x]) for x in self.inkey]
        cols = self.events.ID.astype(int).values.tolist() + ['modnum']

        # get travel times at stations for each vmod
        if parallel:  # if use multiprocessing
            out = Parallel(n_jobs=cores)(delayed(go)(vmod, phi, dx, ein, vnum)
                                         for vmod, vnum in vmods)
        else:  # else old fashion way
            out = []
            for vmod, vnum in vmods:
                out.append(go(vmod, phi, dx, ein, vnum))
        # wrange into travel time df with mod num as index and eid as column
        df = pd.DataFrame(out, columns=cols)
        df.set_index('modnum', drop=True, inplace=True)
        if output == 'travel_times':  # return only travel times
            return df
        if output == 'residuals':
            # subtract phase times from travel times for residuals
            pp = self._get_tt_vector(df)
            df = -df + pp
            return df
        else:
            msg = 'output of %s not supported' % output
            raise ValueError(msg)

    def _get_tt_vector(self, df):
        # function to get a vector of observed phases for subtracting from TT
        dft = pd.DataFrame(columns=df.columns)
        dft2 = self.phases[['EID', 'TT']].T
        dft2.columns = dft2.loc['EID']
        dft.loc['TT'] = dft2.loc['TT']
        return dft.values[0]

    def _get_phi(self, evrow, grid=None):
        """
        Return the phi array for the skfmm
        """
        if grid is None:
            grid = self
        phi = np.ones_like(grid.X[0])
        # index of event in grid
        ind = [int(evrow[x]) for x in self.inkey]
        phi[tuple(ind)] = 0  # set event location to 0
        return phi

    def _get_ray_paths(self, starow, tt, everow, num_blocks):
        # function to back calculate ray paths from station to source
        trail = []  # list to append each coordinate along path
        turPoints = []
        di = {}
        # index of of station in grid
        staind = [int(starow[x]) for x in ['Xin', 'Yin', 'Zin'] if x in starow]
        # make array of infinity to pad travel time boundaries (bounded tt)
        ttb = np.ones(np.add(np.shape(tt), num_blocks)) * np.inf
        # fill in travel times in bottom corner
        ttb[[slice(x, y) for x, y in zip([0] * len(tt.shape), tt.shape)]] = tt
        startTime = tt[tuple(staind)]  # start Time
        timec = startTime  # current time
        cind = tuple(staind)  # current index
        relInd1 = [None]
        turPoints.append(self.ind2coords(cind))
        while timec > 0:
            trail.append(cind)
            cind, relInd2 = self._find_next_block(cind, ttb, num_blocks)
            if any(relInd1) and any(relInd2 / float(max(abs(relInd2))) !=
                                                    relInd1 / float(max(abs(relInd1)))):
                turPoints.append(self.ind2coords(cind))
            relInd1 = relInd2
            timec = ttb[cind]
        turPoints.append(self.ind2coords(cind))
        trail.append(cind)
        di = {'station': starow.ID, 'event': everow.ID, 'startTime': startTime,
              'turningPoints': turPoints}
        return pd.Series(di)

    def _find_next_block(self, cind, ttb, num_blocks):
        neindex = [[int(x + y) for y in range(-num_blocks, num_blocks + 1)]
                   for x in cind]  # index of all neighbors to slice
        # slice of tt with all neibhors within num_blocks in any dim to ind
        neiblock = ttb[np.ix_(*neindex)] - ttb[tuple(cind)]
        for nin, value in np.ndenumerate(neiblock):  # for index (nin) and value of slice
            # index minus number of neighbors, centers 0 on active block
            ni1 = np.array(nin) - num_blocks
            norma = np.multiply(ni1, self.dx)  # normalize for varying lengths
            # RMS of distance from active block
            normfact = np.sqrt(np.sum(np.square(norma)))
            if value != 0.:
                neiblock[nin] = value / normfact
        relativeInd = np.add(np.unravel_index(neiblock.argmin(), neiblock.shape),
                             -num_blocks)  # the relative index of block selected in slice
        argmin = cind + relativeInd  # the index of the block with the least
        return tuple(argmin), relativeInd

    ### Kernel creation and associated functions
    # Note : differently spaced velocity grid from forward problem allowed

    def make_kernel(self, velocityDownSample=5):
        """
        Function to create the output sparse matrix of travel times and
        locations for inverse problem
        Parameters
        ----------
        velocityDownSample : int
            A multiplier to the dx to get an output grid for inversion
        """
        # Check velocityDownSample inputs
        if not isinstance(velocityDownSample, int):
            msg = 'velocityDownSample must be an int'
            raise TypeError(msg)
        # make sure velocityDownSample divides evenly
        if any([len(x) % velocityDownSample != 0 for x in self.xls]):
            msg = (('velocityDownSample %d is not a factor of all dimensions '
                    '%s of xls') % (velocityDownSample, str([len(x) for x in
                                                             self.xls])))
            raise ValueError(msg)

        if len(self.rays) < 1:
            msg = ('fast marching has not yet been called or ray paths not '
                   'calculated. Calculating forward problem.')
            print(msg)
            self.fast_march()

        # get coords of new grid system (for outputs)
        params = self._mesh_grid(self.xmin, self.xmax, list(np.array(self.dx) *
                                                            float(velocityDownSample)), None)
        self.vX, self.vxls, self.vxmax, self.vdx, self.vxnum = params

        if velocityDownSample == 1:  # if no down sampling
            self.kernVel = self.velMod
        else:
            self.kernVel = zoom(self.velMod, 1. / np.array(velocityDownSample),
                                order=0)  # kernel velocity model (downsampled)

        # Get edges
        edges = [list(np.array(self.vxls[num]) - self.vdx[num] / 2.)
                 for num in range(len(self.vdx))]
        for num, a in enumerate(edges):
            a.append(a[-1] + self.vdx[num])
        kernel = []
        for num, row in self.rays.iterrows():  # it each station event pair
            kernel.append(self._get_paths(self.vxls, row, self.vdx, edges))
        return np.array(kernel)

    def _get_paths(self, vxls, row, vdx, edges):
        kernrow = scipy.zeros([len(x) for x in vxls])  # a row of the kernel
        turningPoints = row.turningPoints
        for tnum in range(len(turningPoints) - 1):
            coords = []  # coordinates of each step
            mods = []  # index modifiers of each step
            st = turningPoints[tnum]  # current start point
            stind = self._find_closest_index(st, vxls)  # closest index to start
            ed = turningPoints[tnum + 1]  # current end point
            edind = self._find_closest_index(ed, vxls)  # closest index to end
            dif = np.array(ed) - np.array(st)  # vector of line segement
            inddif = np.array(edind) - np.array(stind)  # vector of index movement
            # if line segment in same block, add len and cont to next turning point
            if stind == edind:
                kernrow[tuple(stind)] += norm(np.array(ed) - np.array(st))
                continue
            coords.append(st)
            mods.append(stind)
            # jacobian of line segment
            J = [[float(x) / y if y != 0 else 0 for x in dif] for y in dif]
            J = list(self._get_unique_rows(J))
            for jnum, j in enumerate(J):
                # if zero jacobian row (IE line doesnt move in this dimension)
                if all(np.array(j) == 0):
                    continue
                sign = np.sign(inddif[jnum])  # sign of step
                firstStep = abs(edges[jnum][int(stind[jnum] + .5 + .5 * sign)]
                                - st[jnum]) * sign
                # which index ticks up for this jacobian
                indtick = np.array([int(x == 1) for x in j]) * sign
                if firstStep != 0:
                    coords.append(np.array(st) + (np.array(j) * firstStep))
                    mods.append(indtick)
                for move in range(1, abs(inddif[jnum])):
                    coords.append(coords[-1] + np.array(j) * np.array(vdx[jnum]) * sign)
                    mods.append(indtick)
            ar = np.hstack([coords, mods])
            # add end coord to end of array
            ar = np.vstack([ar, np.concatenate((ed, np.zeros(len(st))))])
            sortindex = np.lexsort(ar[:, :len(vdx)].transpose())
            # sorted array with [coords,indexmods], only unique rows
            arsrt = self._get_unique_rows(ar[sortindex])
            endar = np.array([all(x == ed) for x in arsrt[:, :len(ed)]]).argmax()
            if endar == 0:  # if end actually occurs at begining switch order
                arsrt = arsrt[::-1]
                endar = np.array([all(x == ed) for x in arsrt[:, :len(ed)]]).argmax()
            arsrt = arsrt[:endar + 1]  # trim to end point
            norms = [scipy.linalg.norm(x) for x in (arsrt[1:, :len(st)] -
                                                    arsrt[:-1, :len(st)])]  # length of each line segment
            gridindex = np.cumsum(arsrt[:, len(st):], axis=0)[:-1]
            for indnum, gridind in enumerate(gridindex):
                grin = tuple([int(x) for x in gridind])  # convert to tuple and ints
                kernrow[grin] += norms[int(indnum)]
        return np.ravel(kernrow)  # returned a flattened version of the kernal row

    def _find_closest_index(self, point, xls):
        return [abs(point[num] - xls[num]).argmin()
                for num in range(len(xls))]  # index of best fit x point in L1 sense

    def _get_unique_rows(self, a):
        # Method for getting unique rows of a np array and preserving order
        df = pd.DataFrame(np.array(a)).drop_duplicates()
        df = df.convert_objects(convert_numeric=True)  # make sure dtypes are right
        return np.array(df)

    ### Get residuals

    def calc_residuals(self):
        """
        Get the residuals from the phases at each station
        """
        if not len(self.phases):  # no phases, cant calc rms
            return None

        if not self.tts:
            self.fast_march(get_rays=False)
        # get index for each station
        sinds = [tuple(self.stations[x]) for x in self.inkey]
        # get travel time to each station for each event, stuff into df add SID
        stimes = [x[sinds] for key, x in self.tts.items()]
        sti_df = pd.DataFrame(stimes, columns=self.stations.ID, index=['pTT'])
        sti_df = sti_df[self.phases.SID].T
        sti_df['SID'] = sti_df.index
        # merge into current phases
        self.phases = self.phases.merge(sti_df)
        # calc residuals and set _rms attr
        self.phases['resid'] = self.phases.TT - self.phases.pTT
        self._rms = norm(self.phases.resid) / np.sqrt(len(self.phases))

    ### Visualization methods

    def plot_contours_2D(self, rayPlotStr='k'):

        """
        Make a 2d plot of contours and calcualted ray paths

        Parameters
        ----------
        rayPlotStr : str
            A plot string recognizable by matplot lib for plotting ray paths
        """

        if len(self.tts) < 1:
            raise Exception('No travel times calculated, run fast_march')

        for ttnum, tt in self.tts.items():
            gs = gridspec.GridSpec(2, 2, width_ratios=[10, .5],
                                   height_ratios=[10, .5], wspace=.05, hspace=.05)
            ax = plt.subplot(gs[0])
            cmesh = ax.pcolormesh(self.xls[0], self.xls[1], self.velMod.transpose(),
                                  cmap='rainbow', alpha=.2)

            cs = ax.contour(self.X[0], self.X[1], tt)
            ax.plot(self.stations.Xap, self.stations.Yap, 'v', lw=500)
            ax.plot(self.events.Xap, self.events.Yap, 'o', lw=500)
            ax.xaxis.tick_top()
            plt.gca().invert_yaxis()
            ax1 = plt.subplot(gs[1])
            cb = plt.colorbar(cs, cax=ax1, label='Travel Time (s)')

            ax2 = plt.subplot(gs[2])
            plt.colorbar(cmesh, cax=ax2, orientation='horizontal',
                         label='Velocity (m/s)')  # ,cax=caxvel)

            for num, row in self.rays.iterrows():
                ax.plot(np.array(row.turningPoints)[:, 0],
                        np.array(row.turningPoints)[:, 1], rayPlotStr, lw=4)

            for line in cb.lines:  # change width of each line
                line.set_linewidth(20)
            ax.set_aspect('equal')
            plt.show()

    ### Misc. functions

    def ind2coords(self, inds):
        """
        Convert index (of forward problem) to spatial coordinates
        """
        return [self.xls[num][x] for num, x in enumerate(inds)]

    def convert_1D(self, vels, deps):
        """
        Convert a 1D vmod with velocities and depths to a grid of the
        appropriate size velocities and depths are in KM!!!!
        """
        # deps = [self.dmin] + list(deps) + [self.dmax]
        mod = np.ones_like(self.X[0]) * vels[0]
        for vnum, vel in enumerate(vels):
            con1 = self.X[-1] > deps[vnum] * 1000.
            con2 = self.X[-1] <= deps[vnum + 1] * 1000.
            mod[con1 & con2] = vel * 1000.
        return mod

    def station_fast_march(self, recalculate=False):
        """
        Generate station travel time grids. These essentially treat the
        station as the source and calculate travel time grids to each
        possible event location
        Parameters
        ----------
        recalculate : bool
            If True recalculate travel time grides if already calculated

        Returns
        -------

        """
        stas = set(self.stations.ID)
        # try loading the disk store
        self.load_hdf5_store()
        # if station grids already there skip
        if stas.issubset(self.station_tts) and not recalculate:
            return
        # iterate through the stations and calc travel time grid
        for ind, row in self.stations.iterrows():
            phi = self._get_phi(row)
            tts = skfmm.travel_time(phi, self.velMod, dx=self.dx)
            self.station_tts[row.ID] = tts
        # if old store is present delete it
        if os.path.exists(self.hdf5_path):
            os.remove(self.hdf5_path)
        # create a store
        self.create_hdf5_store(store_type='station')

    def load_hdf5_store(self, store_type='station'):
        """Function to load the hdf5 cache (if it exists) into memory"""

        if os.path.exists(self.hdf5_path) and self.hdf5_path is not None:
            with tables.open_file(self.hdf5_path, 'r') as f:
                for ind, sta in self.stations.iterrows():
                    name = self._get_hdf5_name(sta.ID, store_type)
                    tts = self._load_hdf5_array(name, f)
                    self.station_tts[sta.ID] = tts


    def create_hdf5_store(self, store_type='station'):
        """Saves travel time grides to disk"""
        if not os.path.exists(self.hdf5_path) and self.hdf5_path is not None:
            with tables.openFile(self.hdf5_path, 'w') as f:
                for ind, sta in self.stations.iterrows():
                    name = self._get_hdf5_name(sta.ID, store_type)
                    tts = self.station_tts[sta.ID]
                    self._write_hdf5_array(name, tts, f)

    def _write_hdf5_array(self, name, tts, f):
            atom = tables.Atom.from_dtype(tts.dtype)
            # save with compression
            #filters = tables.Filters(complib='blosc', complevel=5)
            # ds = f.createCArray(f.root, 'all_data', atom, all_data.shape,
            #  filters=filters)
            # save w/o compression
            ds = f.createCArray(f.root, name, atom, tts.shape)
            ds[:] = tts

    def _load_hdf5_array(self, name, f):
        tts = getattr(f.root, name)[:]
        return tts

    def _get_hdf5_name(self, id, otype):
        """
        Get the name of the expected chunk in the pytables data store
        Parameters
        ----------
        id : int, float, or str
            The ID of the event or station
        otype : str (station or event)
            The type of object to generate a name for
        Returns
        -------
        The str of the expected name
        """
        if otype == 'station':
            pre = 'sta_'
        elif otype == 'event':
            pre = 'eve_'
        else:
            msg = '%s is not a supported otype' % otype
            raise ValueError(msg)
        return pre + '%s' % id

    def locate(self):
        """
        Simple location algorithm to locate the events found in the phases
        file
        Returns
        -------
        A dataframe with estimated locations
        """
        # if station tt grids aren't caclulated do it
        if not self.station_tts:
            self.station_fast_march()
        # concate into 3 or 4 d array
        ar = np.concatenate([x[..., None] for x in self.station_tts.values()],
                            axis=len(self.X))
        # demean
        ar = ar - ar.mean(axis=-1, keepdims=True)
        # arrange arrays




def _flatten_keys(cha):
    """Truncate the keys of the class values, shadow with new values
    on instance to only be 2D"""
    for key in ['spakey', 'apkey', 'inkey']:
        trunc = getattr(cha, key)[:2]
        setattr(cha, key, trunc)
    return cha


class ChaboDriver:
    """
    A meta-chabo, primarily for doing velocity model work. Can be used
    to squish a 3d chabo into 2D when doing 1D model work
    """

    def __init__(self, cha, parallel=True, cores=-1, squish=True):
        """
        Init a ChaboDriver, requires a base chabo instance
        Parameters
        --------
        cha : instance of Chabo
            The base case chabo instance
        parallel : bool
            If True use joblib for multiprocessing
        cores : int ot None
            The number of cores for multiprocessing stuff, -1
            means use all available
        Squish : bool
            If True squish a 3D problem to 2D
        """
        self.parallel = parallel
        self.cores = cores
        if squish:  # flatten the keys if they are to be squished
            skey = self._flatten_stakey(cha.stations)
            ekey = self._flatten_evekey(cha.events, cha.stations)
            # set the spatial keys to 2d version
            cha = _flatten_keys(cha)
        else:  # not tested
            skey = cha.stations
            ekey = cha.events
        phkey = cha.phases
        # init chabos for each station
        self.chas = {}
        for sid in skey.ID.unique():
            # get event key and station key for this station only
            stas = skey[sid == skey.ID]
            eves = ekey[sid == ekey.SID]
            if phkey is not None:
                phs = phkey[phkey.SID == sid]
            else:
                phs = None
            grid = self._make_new_grid(stas, eves, cha)
            cha_new = Chabo(xmin=grid.xmin, xmax=grid.xmax, dx=grid.dx,
                            stations=stas, phases=phs, events=eves)
            self.chas[sid] = cha_new

    def iter_models(self, modgen, output='residuals'):
        """
        Make a panel of residual times (or travel times)
        Parameters
        ----------
        modgen : callable
            A callable that returns an iterable of models,
            model should be something that can attach with the
            chabo.attach_velocity_model function
        output : str
            The type of output to return in the pannel (residuals or
            travel_times)
        """
        dfd = {}  # a data frame dict for storing results
        for sid, cha in self.chas.items():
            models = yield_models(modgen, cha)
            dfd[sid] = cha.iter_models(models, output=output, cores=self.cores,
                                       parallel=self.parallel)
        return pd.Panel(dfd)

    def evaluate_models(self, pan, func):
        """
        Evaluate the models using func contained in the pandas panel produced
        by running self.iter_models, returns a series with model numbers
        as keys and the result of func as values
        Parameters
        ----------
        pan : pd.Panel
            The panel produced by the iter_models method
        func : callable
            A callable that operates on a dataframe of events (index) and
            stations columns. Must handle NaNs as they do appear in the DFs
        """
        from joblib import Parallel, delayed
        # make a generator that slices the panel along the model num axis
        models = pan.major_axis
        df_gen = (pan.loc[:, x, :] for x in models)
        cores = self.cores
        # test
        if self.parallel:
            out = Parallel(n_jobs=cores)(delayed(func)(df) for df in df_gen)
        else:
            out = []
            for df in df_gen:
                out.append(func(df))
        ser = pd.Series(out, models)
        return ser

    def _flatten_stakey(self, skey):
        """
        Function to reduce a station key from 3 d to 2d. It doesn this by
        setting the X to 0 and Y to depth. Then, the events will each be
        smashed relative to the stations
        """
        # make backups
        skey['X_old'], skey['Y_old'], skey['Z_old'] = skey.X, skey.Y, skey.Z
        skey['X'] = 0.0
        skey['Y'] = skey.Z
        return skey

    def _flatten_evekey(self, evkey, stakey):
        """
        Function to reduce the events from 3d to 1D. In order to do this each event
        must have different locations for each station. The horizontal distance
        between each station and the event is calculated and set as the X coord
        The depth is set as the Z coord. An extra column is added for the SID so that
        the event used in each station can be identified later
        """
        evkey['X_old'], evkey['Y_old'], evkey['Z_old'] = evkey.X, evkey.Y, evkey.Z
        out_list = []
        # iterate over each station and create events for it
        for snum, sta in stakey.iterrows():
            # copy df for each station, adjust coords
            df = evkey.copy()
            df['SID'] = sta.ID
            hdist = np.linalg.norm([sta.X_old - df.X, sta.Y_old - df.Y], axis=0)
            df['X'] = hdist
            df['Y'] = df.Z
            out_list.append(df)
        return pd.concat(out_list, ignore_index=True)

    def _make_new_grid(self, sta, eves, cha):
        """
        Function to make a new grid with only the necesary limits
        """
        gr = namedtuple('Grid', 'X, xls, xmax, dx, xnum xmin')
        xmin = [np.min(np.append(eves[col], sta[col]))
                for col in cha.spakey]
        xmax = [np.max(np.append(eves[col], sta[col]))
                for col in cha.spakey]
        xnum = cha.xnum
        dx = cha.dx[:len(cha.spakey)]
        X, xls, xmax, dx, xnum = cha._mesh_grid(xmin, xmax, dx, None)
        return gr(X, xls, xmax, dx, xnum, xmin)


########## Functions for model evaluation
def eval_rms(df):
    """
    Function to evaluate rms of dataframe
    """
    d = df ** 2  # square
    out = d.mean(skipna=True).mean(skipna=True)  # double mean
    # out = d.median(skipna=True).median(skipna=True) # double median
    if np.isnan(out):  # if  NaN is produced return inf so model isn't selected
        return np.inf
    else:
        return np.sqrt(out)


def eval_l1_norm(df):
    """
    Function to evaluate L1 norm of dataframe
    """
    # df.mean(skipna=True, axis=1).hist()
    # plt.show()
    # pudb.set_trace()
    d = abs(df)  # absolute value
    out = d.mean(skipna=True).mean(skipna=True)  # double mean
    # out = d.median(skipna=True).median(skipna=True) # double median
    # d.fillna(np.nan, inplace=True)
    # # ar = d.values.reshape(-1, 1)
    # # ar = ar[~np.isnan(ar)]
    # pudb.set_trace()
    if np.isnan(out):  # if  NaN is produced return inf so model isn't selected
        return np.inf
    else:
        return out


def yield_models(mod_gen, cha):
    """ Yield models from a modgen, should be vels and deps or
    same shape as current velMod"""
    models = mod_gen()
    for mod in models:
        con1 = hasattr(mod[0], 'shape') # duck type
        con2 = np.shape(mod[0]) == np.shape(cha.X[0])
        if con1 and con2:
            yield mod[0], mod[1]
        else:
            (vels, deps), mod_num = mod
        yield cha.convert_1D(vels, deps), mod_num


def _get_vels_deps(x):
    """
    Convert a 1d model with [vels, deps] to a vels and depths array
    """
    dim = len(x) // 2 + 1
    vels = x[:dim]
    if dim == 1:
        deps = ()
    else:
        deps = x[dim:]
    return vels, deps


def go(vmod, phi, dx, sind, modnum):
    """
    Function called by iter_velocities, run in parallel
    """
    tt = skfmm.travel_time(phi, vmod, dx)
    return np.append(tt[sind], modnum)


def init_chabo_on_directory(directory, dx=None, xnum=None):
    """
    Init a chabo instance by reading the files in a directory, finding
    those that are chabo files, setting x, y, and z limits based on those
    files.

    Parameters
    ----------
    directory : str
        A path to the directory
    dx : None, int, float, or list of int, float
        The grid spacing, in meters. If a single number apply to all
        dimensions, else list must be the same length as the dimensions
    xnum : None, int, float, or list of int, float
        The number of grid cells to have in each direction
    Returns
    -------

    """



