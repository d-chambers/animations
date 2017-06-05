#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:39:47 2017

@author: isti_ew
"""

import contextlib

import os

import matplotlib.pyplot as plt
import numpy as np
import skfmm
import xarray as xr
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns

# directory path for giffs
JPEG_DIR = 'jpegs'

# define basic geometry
DX = 50
X = np.arange(0, 2010, DX)
Y = np.arange(0, 2010, DX)
Z = np.arange(0, 2010, DX)

ORIGIN = (1100, 1000, 1000)  # location of event origin
# velocity in the z coordinate
BACKGROUND_VELOCITY = 6000
VELOCITIES = [(1100, 5000), (1400, 4000), (1700, 2500)]

STOPE_COORDS = [(1000, 1050), (600, 1400), (600, 1400)]
STOPE_VELOCITY = 500

# a color map based on velocity

COLOR1 = tuple(np.array([255, 10, 10])/256.)
COLOR2 = tuple(np.array([10, 255, 10])/256.)
COLOR3 = tuple(np.array([10, 10, 255])/256.)
COLOR_MAP = {VELOCITIES[0][1]: COLOR1, VELOCITIES[1][1]: COLOR2, 
             VELOCITIES[2][1]: COLOR3, STOPE_VELOCITY: '0.2'}

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
#    ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
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
    

def plot_orb(dist: xr.DataArray, contour: float, ax=None, **kwargs):
    """ plot the expanding wavefront """

    
    ax = plt.figure().add_subplot(111, projection='3d') if ax is None else ax
    plot_surface(ax, dist.values, contour, **kwargs)
    


def plot_surface(ax, ar: np.array, contour_value: float, **kwargs):
    verts, faces = measure.marching_cubes(ar, contour_value)
    alpha = kwargs.pop('alpha', .2)
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], alpha=alpha,
                    **kwargs)


def plot_dot(ax):
    """ plot a dot where the event originates """
    plt.scatter([ORIGIN[0]], [ORIGIN[1]], [ORIGIN[2]])

# ------------------------- Run animations
# make gif directory
if not os.path.isdir(JPEG_DIR):
    os.mkdir(JPEG_DIR)

# calculate travel times, get speed vecto with stopes
speed_with_stope = set_stope(speed)
travel_times = get_travel_times(phi, speed_with_stope)

NUM_FRAMES = 125

# set travel times
ls = np.linspace(travel_times.min(), travel_times.max(), NUM_FRAMES + 2)

# set azimuth and elevation params
azimuths = np.linspace(20, 360, NUM_FRAMES)
elevations = np.linspace(travel_times.Z.min(), travel_times.Z.max())

for num, tt in enumerate(ls[1:-1]):
    with get_fig_and_axis(speed) as (f, ax):
        # plot velocity layers
        plot_velocity_changes(speed, ax, alpha=.15)
        # plot stope
        plot_surface(ax, speed_with_stope.values, STOPE_VELOCITY, color='0.2',
                     alpha=.3)
        plot_orb(travel_times, tt, ax=ax)
        # plot the origin of the events
        plot_orb(travel_times, .007, ax=ax, alpha=1, color='k')
#        plot_dot(ax)
        ax.view_init(azim=azimuths[num], elev=8)
    plt.show()
    path = os.path.join(JPEG_DIR, f'{num:03d}.jpeg')
#    plt.subplots_adjust(left=0.01, right=0.011, top=0.011, bottom=0.01)
#    ax.set_facecolor('.75')
    f.set_size_inches(8, 8, forward=True)
    f.tight_layout()
    f.savefig(path, dpi=300)
    
    
#ffmpeg -r 10 -f image2 -i %03d.jpeg -qscale 0 fastmarch.mp4
    
    