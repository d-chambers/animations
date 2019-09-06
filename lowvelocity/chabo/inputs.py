# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:15:55 2015

@author: isti_ew
"""
from __future__ import division, unicode_literals, absolute_import, print_function
from six import string_types

import numpy as np
import pandas as pd
from obspy.core import UTCDateTime as utc
import sys

###FUNCTIONS FOR GENERATING STATION LIST
stypes=['obspy', 'hypoinv', 'hypoel', 'hypodd', 'seisan', 'nll', 'df']
def _checkStationInput(sfile,stype): #make sure event format and type are kosher
    if not stype in stypes:
        raise Exception ('Station file format not supported. Supported types are %s'%str(stypes) )
    if stype == 'df' and not isinstance(sfile,pd.DataFrame):
        raise TypeError ('Expected event file input is a dataframe')
    elif not stype == 'df' and not type(sfile) == str: #Is there such a thing as a path object that this should also allow?
        raise TypeError ('Expected event file input is a path')

def _getStations(sfile,stype,Grid,**kwargs):
    if stype=='obspy': 
        stations=_makesobspy(sfile,Grid,**kwargs)  
    elif stype=='df': 
        stations=_makesdf(sfile)
    elif stations=='hypoinv':
        raise Exception('Need to create logic for this station file format')
    elif stations=='hypoel':
        raise Exception('Need to create logic for this station file format')
    elif stype=='hypodd':
        raise Exception('Need to create logic for this station file format')
    elif stype=='seisan':
        raise Exception('Need to create logic for this station file format')
    elif stype=='nll':
        raise Exception('Need to create logic for this station file format')
    else: #Shouldn't be possible
        raise Exception ('Station file type not supported. Supported types are %s'%str(stypes))       
    return stations
    
### make events functions   
def _makesdf(sfile):
    #This just checks that the required columns exist in the data frame
    stations = sfile
    msg = "Required columns for the phase file are ID, X, Y, Z"
    if not 'ID' in stations.columns:
        raise Exception(msg)
    if not 'X' in stations.columns:
        raise Exception(msg)
    if not 'Y' in stations.columns:
        raise Exception(msg)
    if not 'Z' in stations.columns:
        raise Exception(msg)      
    return stations

def _makesobspy(sfile,Grid,**kwargs):
    stations = pd.read_csv(sfile)
    stations = _makesdf(stations)
    return stations
    
def Stations(sfile, Grid, stype='df', **kwargs):
    """
    Function for generating events\n
    Parameters
    ----------
    efile : event file
        Various types supported
    
    Grid : Instance of Gridded class 
        Chabo grid to attach the phases to
    
    etype : str
        string of event file format
    
    """
    _checkStationInput(sfile,stype)
    stations = _getStations(sfile,stype,Grid,**kwargs)
    return stations

###FUNCTIONS FOR GENERATING EVENT LIST
etypes=['obspy','hypoinv','hypoel','hypodd','seisan','nll','df']
def _checkEventInput(efile,etype): #make sure event format and type are kosher
    if not etype in etypes:
        raise Exception ('Event file format not supported. Supported types are %s'%str(etypes) )
    if etype == 'df' and not isinstance(efile,pd.DataFrame):
        raise TypeError ('Expected event file input is a dataframe')
    elif not etype == 'df' and not type(efile) == str: #Is there such a thing as a path object that this should also allow?
        raise TypeError ('Expected event file input is a path')

def _getEvents(efile,etype,Grid,**kwargs):
    if etype=='obspy': 
        events=_makeeobspy(efile,Grid,**kwargs)  
    elif etype=='df': 
        events=_makeedf(efile)
    elif etype=='hypoinv':
        raise Exception('Need to create logic for this phase file format')
    elif etype=='hypoel':
        raise Exception('Need to create logic for this phase file format')
    elif etype=='hypodd':
        raise Exception('Need to create logic for this phase file format')
    elif etype=='seisan':
        raise Exception('Need to create logic for this phase file format')
    elif etype=='nll':
        raise Exception('Need to create logic for this phase file format')
    else: #Shouldn't be possible
        raise Exception ('Event file type not supported. Supported types are %s'%str(etypes))       
    return events
    
### make events functions   
def _makeedf(events):
    #This just checks that the required columns exist in the data frame
    req_cols = {'ID', 'O', 'X', 'Y', 'Z', 'EType'}
    cols = events.columns
    msg = "Required columns for the event file are ID, O, X, Y, Z, EType"
    if not req_cols.issubset(cols):
        msg2 = '. Missing columns are %s' % (req_cols.difference(cols))
        raise ValueError(msg + ' ' + msg2)
    return events

def _makeeobspy(efile,Grid,**kwargs):
    events = pd.read_csv(efile)
    events['UTC']=[utc(x) for x in events['O']]
    events.drop('O',axis=1,inplace=True)
    events.rename(columns={'UTC':'O'},inplace=True)
    events = _makeedf(events)
    return events
    
def Events(efile,Grid,etype='df',**kwargs):
    """
    Function for generating events\n
    Parameters
    ----------
    efile : event file
        Various types supported
    
    Grid : Instance of Gridded class 
        Chabo grid to attach the phases to
    
    etype : str
        string of event file format
    
    """
    _checkEventInput(efile,etype)
    events = _getEvents(efile,etype,Grid,**kwargs)
    return events

###FUNCTIONS FOR GENERATING PHASE LIST
ptypes=['obspy','hypoinv','hypoel','hypodd','seisan','nll','df']
def _checkPhaseInput(pfile,ptype): #make sure phase format and type are kosher
    if not ptype in ptypes:
        raise Exception ('Phase file format not supported. Supported types are %s'%str(ptypes) )
    if ptype == 'df' and not isinstance(pfile, pd.DataFrame):
        raise TypeError ('Expected phase file input is a dataframe')
    elif not ptype == 'df' and not isinstance(pfile, string_types): #Is there such a thing as a path object that this should also allow?
        raise TypeError ('Expected phase file input is a path')

def _getPhases(pfile,ptype,Grid,**kwargs): 
    if ptype == 'obspy': 
        phases = _makepobspy(pfile, Grid, **kwargs)  
    elif ptype == 'df': #Need to slightly rework to have tt not arrival time
        phases = _makepdf(pfile)
    elif ptype == 'hypoinv':
        raise Exception('Need to create logic for this phase file format')
    elif ptype == 'hypoel':
        raise Exception('Need to create logic for this phase file format')
    elif ptype == 'hypodd':
        raise Exception('Need to create logic for this phase file format')
    elif ptype == 'seisan':
        raise Exception('Need to create logic for this phase file format')
    elif ptype == 'nll':
        raise Exception('Need to create logic for this phase file format')
    else: #Shouldn't be possible
        raise Exception ('Phase file type not supported. Supported types are %s'%str(ptypes))           
    return phases
    
### make phases functions   
def _makepdf(pfile):
    #This just checks that the required columns exist in the data frame
    req_cols = set(['EID', 'SID', 'P', 'W', 'TT'])
    phases = pfile
    msg = "Required columns for the phase file are: %s" % req_cols
    if not req_cols.issubset(phases.columns):
        raise ValueError(msg)       
    return phases

def _makepobspy(pfile,Grid,**kwargs):
    phases = pd.read_csv(pfile)
    phases['UTC']=[utc(x) for x in phases['Time']]
    phases['TT']=np.nan
    for num,row in phases.iterrows():
        event = Grid.events[Grid.events['ID']==row['EID']]
        station = Grid.stations[Grid.stations['ID']==row['SID']]
        if len(event) == 1 and len(station) == 1:
            #Compute the travel time
            phases.loc[num,'TT'] = row['UTC']-event.loc[0]['O']
        else:
            #Toss the phase
            phases.drop(num,inplace=True)
    _makepdf(phases)
    return phases
        
def Phases(pfile,Grid,ptype='df',**kwargs):
    """
    Function for generating phases\n
    Parameters
    ----------
    pfile : phase file
        Various types supported
    
    Grid : Instance of Gridded class 
        Chabo grid to attach the phases to
    
    ptype : str
        string of phase file format
    
    """
    _checkPhaseInput(pfile,ptype)
    phases = _getPhases(pfile, ptype, Grid, **kwargs) #initialize phases
    return phases