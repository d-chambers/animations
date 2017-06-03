#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:12:55 2017

@author: isti_ew
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PatchCollection

import seaborn as sns

STARTING_POINTS = np.array([[-.25, -.25], [0, 1], [1, 0]])


# STARTING_VALUES


def func(point):
    """ The Rosenbrock function with a=0, b=1:
    https://en.wikipedia.org/wiki/Rosenbrock_function """
    x = point[0]
    y = point[1]
    return x ** 2 + (y - x ** 2) ** 2


class Simplex:
    """ class to capture behavior of simplex for graphing """

    alpha = 1  # reflection coef.
    gamma = 2  # expansion coef.
    rho = 0.5  # contraction coef.
    sigma = 0.5  # shrink coef

    def __init__(self, points, values_0=None, func=None):
        # asserts and setups
        assert values_0 is not None or func is not None
        if values_0 is None:  # evaluate
            values_0 = np.array([func(x) for x in points])
        assert len(points) == len(values_0)
        # value place-holder
        self.cvalues_ =values_0.astype(float)
        self.cpoints = points.astype(float)  # current points
        self.ppoints = points.astype(float)  # previous points

    # -------------------------------- properties
    @property
    def ccentroid(self):
        return calculate_centroid(self.cpoints)

    @property
    def pcentroid(self):
        return calculate_centroid(self.ppoints)

    @property
    def max_vertex(self):
        return self.cpoints[np.argmax(self.cvalues_)]

    @property
    def min_vertex(self):
        return self.cpoints[np.argmin(self.cvalues_)]

    @property
    def reflection(self):
        """ get reflected version of triangle """
        return self.ccentroid + (self.ccentroid - self.max_vertex) * self.alpha

    @property
    def expansion(self):
        """ get the expansion point """
        return self.ccentroid + (self.reflection - self.ccentroid) * self.gamma

    @property
    def contraction(self):
        """ get contraction point """
        return self.ccentroid + (self.max_vertex - self.ccentroid) * self.rho

    @property
    def shrink(self):
        """ get shrunken simplex """
        ar = np.copy(self.cpoints)
        lowest_ind = np.argmin(self.cvalues_)
        for num, subar in enumerate(ar):
            if num == lowest_ind:
                continue
            ar[num] = self.min_vertex + self.sigma * (subar - self.min_vertex)
        return ar

    # -------------------------------- methods
    def update_highest_point(self, new_point):
        ar = np.copy(self.cpoints)
        ar[int(np.argmax(self.cvalues_)), :] = new_point
        assert np.all(ar[np.argmax(self.cvalues_)] == new_point)
        return ar


# ------------------------------------- simplex auxiliary methods

def to_polygon(points, **kwargs):
    """ given points, wrap them in a closed polygon """
    return Polygon(points, closed=True, **kwargs)


def calculate_centroid(array):
    """ given an array of vertices calculate a centroid """
    return array.mean(axis=0)


# ------------------------- matplot lib plotting stuff


def frame_by_frame(points_1, points_2, num=25):
    """ Given two sets of points that represent polygons, return a new
    ndarray (with shape = shape(points_1) + (num)) that extrapolates between
    each point linearly in order to pass to visualization to simulate
    motion """
    assert points_1.shape == points_2.shape
    assert np.any(points_1 != points_2)
    change_vector = (points_2 - points_1) / float(num)
    out = np.zeros(tuple([int(num) + 1] + list(np.shape(points_1))))
    for ind in range(int(num) + 1):
        out[ind] = points_1 + (ind * change_vector)
    return out


#def save_fig(path, **figwargs):
#    """ decorator factory to save figure in path directory """
#    path = os.path.join(os.path.dirname(__file__), path)
#    if not os.path.exists(path):
#        os.mkdir(path)
#
#    def deco(func):
#        nonlocal counter
#        counter = 0
#
#        def wrap(self, *args, **kwargs):
#            f = plt.Figure(**figwargs)
#            func(*args, **kwargs)
#            save_path = os.path.join(path, f'{counter:0.3d}.jpeg')
#            f.savefig(save_path)
#            counter += 1
#
#        return wrap
#
#    return deco


def make_path(path):
    path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(path):
        os.mkdir(path)


class PlotPlex2D:
    """ convenience class for plotting simplex """

    def __init__(self):
        self.simplex = Simplex(STARTING_POINTS, values_0=np.array([1, 0, 0]))
        
    def get_figure(self):
        fig = plt.Figure(figsize=(1, 1))
        return fig
    
    def set_axis_limits(self, ax):
        cent_x = self.simplex.ccentroid[0]
        cent_y = self.simplex.ccentroid[1]
        
        ax.set_xlim(cent_x - 1, cent_x + 1)
        ax.set_ylim(cent_y - 1, cent_x + 1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def plot_2d_reflection(self):
        s1 = self.simplex.cpoints
        s2 = self.simplex.update_highest_point(self.simplex.reflection)
        self._plot_2D_sequence('Reflection', s1, s2)
    
    def plot_2d_expansion(self):
        s1 = self.simplex.cpoints
        s2 = self.simplex.update_highest_point(self.simplex.expansion)
        self._plot_2D_sequence('Expansion', s1, s2)
    
    def plot_2d_contraction(self):
        s1 = self.simplex.cpoints
        s2 = self.simplex.update_highest_point(self.simplex.contraction)
        self._plot_2D_sequence('Contraction', s1, s2)
        
    def plot_2d_shrink(self):
        s1 = self.simplex.cpoints
        s2 = self.simplex.shrink
        self._plot_2D_sequence('Shrink', s1, s2)
        

    def _plot_2D_sequence(self, path, s1, s2, ax=None):
        make_path(path)
        seqs = frame_by_frame(s1, s2)
        centroid = self.simplex.ccentroid
        new_centroid = calculate_centroid(s2)
        for num, seq in enumerate(seqs):
            plt.show()
            fig = self.get_figure()
            plt.axes().set_aspect('equal')
            ax = plt.gca()
            poly = to_polygon(seq)
            collection = PatchCollection([poly], alpha=.3, linewidth=2,
                                         edgecolor='k')
            ax.add_collection(collection)
#            import pdb; pdb.set_trace()
            ax.scatter([centroid[0]], [centroid[1]], color='k')
            ax.scatter([new_centroid[0]], [new_centroid[1]], color='r')
            self.set_axis_limits(ax)
            plt.title(path, fontsize=20)
            fin = f'{num:03d}.png'
            file_name = os.path.join(path, fin)
            plt.savefig(file_name)
            if num == len(seqs) - 1:
                for num in range(num, num + len(seqs) // 2):
                    fin = f'{num:03d}.png'
                    file_name = os.path.join(path, fin)
                    plt.savefig(file_name)


# ----------------------------- Run animations
pp = PlotPlex2D()
pp.plot_2d_reflection()
pp.plot_2d_expansion()
pp.plot_2d_contraction()
pp.plot_2d_shrink()



















