#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:39:47 2017

@author: isti_ew
"""

import skfmm
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy import sin, cos, pi
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D
import xarray as xr
import contextlib
import seaborn as sns

# define basic geometry
DX = 100
X = np.arange(0, 2010, DX)
Y = np.arange(0, 2010, DX)
Z = np.arange(0, 2010, DX)

ORIGIN = (900, 1000, 1000)  # location of event origin
# velocity in the z coordinate
BACKGROUND_VELOCITY = 6000
VELOCITIES = [(800, 5000), (1200, 4500), (1800, 3500)]

STOPE_COORDS = [(1000, 1050), (600, 1400), (600, 1400)]
STOPE_VELOCITY = 500

# a color map based on velocity
COLOR_MAP = {VELOCITIES[0][1]: 'r', VELOCITIES[1][1]: 'b', 
             VELOCITIES[2][1]: 'g', STOPE_VELOCITY: '0.5'}

grid_shape = (len(X), len(Y), len(Z))

# Create the mine initiation array and velocity array
dims = 'X Y Z'.split()
coords = {'X': X, 'Y': Y, 'Z': Z}

phi = xr.DataArray(np.ones(grid_shape), dims=dims, coords=coords, name='phi')
speed = xr.DataArray(np.ones(grid_shape) * BACKGROUND_VELOCITY, coords=coords,
                     dims=dims, name='speed')

# set origin location
origin = phi.sel(X=ORIGIN[0], Y=ORIGIN[1], Z=ORIGIN[2])
phi.loc[origin.coords] = 0

# set layered velocities
for depth, velocity in VELOCITIES:
    speed.loc[{'Z': speed.Z > depth}] = velocity
             
# add stope
def set_stope(speed):
    """ create a copy of the speed array with stopes set """
    dar = speed.copy(deep=True)
    stope_dict = dict()
    for coord, (lower, upper) in zip('X Y Z'.split(), STOPE_COORDS):
        stope_dict[coord] = ((speed.coords[coord] <= upper) & 
                             (speed.coords[coord] >= lower))
    dar.loc[stope_dict] = STOPE_VELOCITY
    return dar


def get_travel_times(phi, speed):
    """ given the initation matrix (phi) and the velocity array, calculate
    travel times """

    tt = skfmm.travel_time(phi.values, speed.values, dx=DX)
    travel_times = xr.DataArray(tt, coords=coords, dims=dims)
    return travel_times


# setup axis and figure

@contextlib.contextmanager
def get_fig_and_axis(speed, figsize=(8, 8)):
    """ get the figure and axis objects, then set outputs """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    yield fig, ax
    # set axis bounds
    ax.set_xbound(0, len(speed.X))
    ax.set_ybound(0, len(speed.Y))
    ax.set_zbound(0, len(speed.Z))
    # set ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.show()
    

def plot_velocity_changes(speed, ax=None, **kwargs):
    """ plott the velocity gradiants """
    ax = plt.figure().add_subplot(111, projection='3d') if ax is None else ax
    uniques = np.unique(speed.values)    
    # remove background velocity
    uniques = uniques[uniques != BACKGROUND_VELOCITY]
    for unique in uniques + 1:
        color = COLOR_MAP.get(unique - 1, None)
        plot_surface(ax, speed.values, unique, color=color, **kwargs)
    

def plot_orb(dist: xr.DataArray, contour: float, ax=None):
    """ plot the expanding wavefront """

    
    ax = plt.figure().add_subplot(111, projection='3d') if ax is None else ax
    plot_surface(ax, dist.values, contour)
    


def plot_surface(ax, ar: np.array, contour_value: float, **kwargs):
    verts, faces = measure.marching_cubes(ar, contour_value)
    alpha = kwargs.pop('alpha', .2)
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], alpha=alpha,
                    **kwargs)


speed_with_stope = set_stope(speed)
travel_times = get_travel_times(phi, speed_with_stope)

ls = np.linspace(travel_times.min(), travel_times.max(), 25)
for tt in ls[1:-1]:
    print(f'plotting {tt}')
        
    with get_fig_and_axis(speed) as (f, ax):
        # plot velocity layers
        plot_velocity_changes(speed, ax, alpha=.15)
        # plot stope
        plot_surface(ax, speed_with_stope.values, STOPE_VELOCITY, color='0.5')
        plot_orb(travel_times, tt, ax=ax)

#    plot_orb(travel_times, .2, ax=ax)






#
#def fun(x, y, z):
#    return cos(x) + cos(y) + cos(z)
#
#x, y, z = pi*np.mgrid[-1:1:31j, -1:1:31j, -1:1:31j]
#vol = fun(x, y, z)
#verts, faces = measure.marching_cubes(vol, 0, spacing=(0.1, 0.1, 0.1))
#


                #cmap='Spectral', lw=0, shade=False)
#plt.show()