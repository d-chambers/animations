# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:56:42 2015

@author: isti_ew
"""
from __future__ import division, absolute_import, unicode_literals, print_function
from six import string_types

import numpy as np
import pandas as pd
from csv import reader
import warnings
import struct

import pdb

class VelocityGrid(object):
    """
    Class for generating velocity grids\n
    Parameters
    ----------
    vel : velocity model
        Various types supported
    
    Grid : Instance of Chabo class 
        Defines problem geometry in local space
    
    vtype : str
        string of velocity model type
    
    """
    vtypes = ['1d', '1dcsv', '2dgrid', '2dcsv', '3dgrid', '3dcsv', '3dnll',
              None]

    def __init__(self, velmod, Grid, vtype=None, **kwargs):
        # Sanity checks
        self._checkVelocityInput(velmod, vtype)
        #if the grid and the passed velocity model have the same shape use it
        if np.shape(Grid.X[0]) == np.shape(velmod) and vtype in ['3dgrid', None]:
            self.velocity = velmod            
        #if velmod is already a VelocityGrid instance, use it
        elif isinstance(velmod, VelocityGrid):
            self = VelocityGrid
        else: #initialize grid
            self.velocity = self._makeVelocity(velmod, vtype, Grid, **kwargs) 
        
    # make sure velocity format and type are kosher
    def _checkVelocityInput(self, velmod, vtype): 
        # make sure the velcity method is supported
        if not vtype in VelocityGrid.vtypes:
            msg = (('Velocity type not supported. Supported types are %s') % 
                    str(VelocityGrid.vtypes)) 
            raise ValueError(msg)
        if vtype is None:
            return

        #This should be more general... 
        if vtype == '1d' and isinstance(velmod, string_types): 
            msg = ('1d velocity model does not support type %s' % type(velmod))
            raise TypeError (msg)
        elif 'csv' in vtype and not isinstance(velmod, string_types): 
            msg = 'csv input: velMod must be path to velocity model'
            raise TypeError (msg)
        if vtype == '2dgrid' and isinstance(velmod, string_types): 
            msg = '2dgrid: Model does not support type %s' % type(velmod)
            raise TypeError (msg)
        if vtype == '3dgrid' and isinstance(velmod, string_types):
            msg = '3dgrid: Model does not support type %s' % type(velmod)
            raise TypeError(msg)
    #Do we want to add some kind of check at the end just to verify that the 
    #resulting velocity grid is the same dimensions as the Chabo grid?
    def _makeVelocity(self, velmod, vtype, Grid, **kwargs):
        # try to determine velocity type if it isnt given
        if vtype == None:
            vtype = self.guess_at_vtype(velmod)

        # set background velocity
        backVel = kwargs.get('backVel', -9.) #background velocity
        velocity = backVel * np.ones_like(Grid.X[0])  # init ones array
#        if not backVel == -99.:
#            backVel = backVel * 1000.
        # global velocity # TODO remove this global if it isnt used

        # go find correct type
        
        if vtype == '1d':
            velocity = self._make1d(velmod, velocity, Grid, **kwargs)           
        elif vtype == '1dcsv':
            velocity = self._make1dcsv(velmod, velocity, Grid, **kwargs) 
        elif vtype == '2dgrid':
            velocity = self._make2dgrid(velmod, velocity, Grid, **kwargs)      
        elif vtype == '2dcsv':
            warnings.warn('csv velocity model must have spacing less than or '
                            'equal to Chabo grid for accurate mapping', Warning)
            velocity = self._make2dcsv(velmod, velocity, Grid, **kwargs)                    
        elif vtype == '3dgrid':
            velocity = self._make3dgrid(velmod, velocity, Grid, **kwargs)
        elif vtype == '3dcsv':
            warnings.warn('csv velocity model must have spacing less than or '
                            'equal to Chabo grid for accurate mapping', Warning)
            velocity = self._make3dcsv(velmod, velocity, Grid, **kwargs)     
        elif vtype == '3dnll':
            velocity = self._make3dnll(velmod, velocity, Grid, **kwargs)
        else: #Shouldn't be possible
            msg = (('Velocity type not supported. Supported types are %s'
                    % str(VelocityGrid.vtypes)))
            raise ValueError (msg)
        if -99 in velocity:
            raise Exception ('Velocity model is not defined over all regions.')
            
        return velocity

    def guess_at_vtype(self, velmod):
        if isinstance(velmod, np.ndarray):
            shape = np.shape(velmod)
            if len(shape) == 2:
                return '1d'
            elif len(shape) == 4:
                return '2dgrid'
            elif len(shape) == 3:
                return '3dgrid'

    ### make velocity functions
    
    def _coord2ind(self, modArray, xls):
        newArray = []
        for point in modArray:
            for num2 in range(len(xls)):
                #index of best fitting x point in L1 sense
                xind = [abs(point[num2]-xls[num2]).argmin() 
                        for num2 in range(len(xls))]        
            newArray.append([tuple(xind), point[len(xls)]])
        return newArray
        
    def _make1d(self, velmod, velocity, Grid, **kwargs):
        ind = kwargs.get('index', -1) # get index keyword else return -1
        for x in velmod:
            velocity[Grid.X[ind] >= x[1] * 1000.] = x[0] * 1000.0    
        return velocity
        
    def _make1dcsv(self, velmod, velocity, Grid, **kwargs):
        with open(velmod, 'rb') as f:
            r = reader(f)
            modInput = np.array(list(r))
        try:
            modInput = modInput.astype(float)
        except TypeError:
            raise TypeError ('Velocity model file can only contain numbers')
        velocity = self._make1d(modInput, velocity, Grid, **kwargs)
        return velocity
        
    def _make2dgrid(self, velmod, velocity, Grid, **kwargs):
        ind = kwargs.get('index',-1) #Get the orientation of the grid
        #Need to reshape the matrix along the index axis
        if len(velmod.shape) < len(Grid.xnum): 
            if ind == 0: #Is there a cleaner way to do this?
                velocity = velmod.reshape(1,velmod.shape[0], velmod.shape[1]
                            ).repeat(len(Grid.xls[ind]), axis=ind)
            elif ind == 1:
                velocity = velmod.reshape(velmod.shape[0], 1, velmod.shape[1]
                            ).repeat(len(Grid.xls[ind]), axis=ind)
            else:
                velocity = velmod.reshape(velmod.shape[0], velmod.shape[1], 1
                            ).repeat(len(Grid.xls[ind]), axis=ind)
        elif velmod.shape == tuple(Grid.xnum): #The grid corect
            velocity = velmod
        elif velmod.shape[ind] == 1:
            velocity = velmod.repeat(len(Grid.xls[ind]), axis = ind)
        else:
            msg = (('Dimensions of velocity model %s do not match dimensions '
                    'of grid %s') % (velmod.shape, Grid.xnum))
            raise TypeError(msg)
        return velocity
            
    def _make2dcsv(self, velmod, velocity, Grid, **kwargs):
        ind = kwargs.get('index', -1) #Get the orientation of the grid
        with open(velmod, 'rb') as f:
            r = reader(f)
            modInput = np.array(list(r))
        try:
            modInput = modInput.astype(float)
        except TypeError:
            raise TypeError ('Velocity model file can only contain numbers')
        axes = [i for i in range(len(Grid.xls)) if not i == ind]
        if ind == -1:
            axes.pop(len(Grid.xls)-1)

        modInput = self._coord2ind(modInput,np.array(Grid.xls)[axes])

        #Take a slice of the grid along the specified axis and apply the 
        #specified velocities to the relevant indices...
        axis = [slice(None)] * len(velocity.shape)
        axis[ind] = slice(0,1)
        velslice = velocity[axis]
        for x in modInput:
            velslice[x[0]] = x[1]
        
            
        #Run the resulting 2D grid through _make2dgrid to get the 3d grid
        velocity = self._make2dgrid(velslice, velocity, Grid, **kwargs)
        return velocity

    def _make3dgrid(self, velmod, velocity, Grid, **kwargs):
        if velmod.shape == tuple(Grid.xnum):
            velocity = velmod
        else:
            msg = (('Dimensions of velocity model %s do not match dimensions '
                    'of grid %s, check vtype') % (velmod.shape, Grid.xnum))
            raise TypeError(msg)
        return velocity
        
    def _make3dcsv(self,velmod,velocity,Grid,**kwargs):
        #ind=kwargs.get('index',-1) #Get the orientation of the grid
        with open(velmod, 'rb') as f:
            r = reader(f)
            modInput = np.array(list(r))
        try:
            modInput = modInput.astype(float)
        except (TypeError, ValueError):
            raise TypeError ('Velocity model file can only contain numbers')

        modInput = self._coord2ind(modInput,np.array(Grid.xls))

        #Apply the specified velocities to the relevant indices
        for x in modInput:
            velocity[x[0]] = x[1]
                    
        # Run the resulting grid through _make3dgrid to get the 3d grid 
        # (mainly to just make sure it has the correct dimensions)
        velocity =self._make3dgrid(velocity,velocity,Grid,**kwargs)
        return velocity

    def _make3dnll(self,velmod,velocity,Grid,**kwargs):
        headname = velmod + ".hdr"
        binname = velmod + ".buf"
        with open(binname,'rb') as binfile:
            binvelocs = []
            for block in iter(lambda: binfile.read(4),""):
                binvelocs.append(struct.unpack('f',block)[0])
        
        # Read in the header and use it to size a grid for the velocity 
        # Will need to either make sure that it matches the existing Chabo grid,
        # or give the user the option to replace the Chabo grid?
        with open(headname,'r') as headfile:
            header = headfile.read().split()
            
        headxnum = [int(header[0]) - 1, int(header[1]) - 1, int(header[2]) - 1]
        headdx = [float(header[6]) * 1000, float(header[7]) * 1000, 
                  float(header[8]) * 1000]
        headxmax = [float(header[3]) * 1000 + headxnum[0] * headdx[0], 
                    float(header[4]) * 1000 + headxnum[1] * headdx[1], 
                    float(header[5]) * -1000]
        
        if not (headxnum == Grid.xnum):
            msg = (('NLL grid xnum %s does not match Chabo grid xnum %s') % 
                    (headxnum,Grid.xnum))
            raise TypeError (msg)
        if not (headxmax == Grid.xmax):
            msg = (('NLL grid xmax %s does not match Chabo grid xmax %s') % 
                    (headxmax,Grid.xmax))
            raise TypeError ()
        if not (headdx == Grid.dx):
            msg = (('NLL grid dx %s does not match Chabo grid dx %s') % 
                    (headdx,Grid.dx))
            raise TypeError (msg)
            
        velocity = np.ones(headxnum)
        i,j,l=0,0,0
        k = headxnum[2]-1
        while i<headxnum[0]:
            while j<headxnum[1]:
                while k>=0:
                    velocity[i,j,k]=binvelocs[l]
                    l = l + 1
                    k = k - 1
                j = j + 1
                l = l + 1
                k = headxnum[2]-1
            i = i + 1
            l = l + headxnum[2] + 1
            j = 0
        return velocity        
    
    ### Initiates
    
    def __repr__(self):
        string = str(self.velocity)
        return string
    
    def __getitem__(self, ind):
        return self.velocity[ind]
    
    #if attribute not found default to np array of velocities
    def __getattr__(self, attr): 
        return self.velocity.__getattribute__(attr)
        
