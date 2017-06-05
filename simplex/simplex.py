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
# import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# STARTING_POINTS = np.array([[-.25, -.25], [0, 1], [1, 0]])
STARTING_POINTS = np.array([[-2, -2], [-1, -1], [-1, -.5]])

# STARTING_VALUES


def func(point):
    """ The Rosenbrock function with a=0, b=1:
    https://en.wikipedia.org/wiki/Rosenbrock_function """
    x = point[0]
    y = point[1]
    return x ** 2 + (y - x ** 2) ** 2


class cache:
    """ A property like cache """
    def __init__(self, method):
        self.method = method

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        assert hasattr(instance, 'cache')
        if self.name not in instance.cache:
            instance.cache[self.name] = self.method(instance)
        return instance.cache[self.name]


class Simplex:
    """ class to capture behavior of simplex for graphing """

    alpha = 2  # reflection coef.
    gamma = 4  # expansion coef.
    rho = 0.5  # contraction coef.
    sigma = 0.5  # shrink coef

    def __init__(self, points, values_0=None, func=None):
        # asserts and setups
        assert values_0 is not None or func is not None
        if values_0 is None:  # evaluate
            values_0 = np.array([func(x) for x in points])
        assert len(points) == len(values_0)
        # value place-holder
        self.cvalues_ = values_0.astype(float)
        self.cpoints = points.astype(float)  # current points
        self.ppoints = points.astype(float)  # previous points
        self.last_move = None
        self.func = func

        self.p_min_values = np.min(self.cvalues_)

        self.cache = {}

    # ---------------------------- cached properties
    @cache
    def sorted_cvalues(self):
        return self.cvalues_[np.argsort(self.cvalues_)]

    @cache
    def sorted_values(self):
        return np.sort(self.cvalues)

    @cache
    def cvalues(self):
        return np.apply_along_axis(self.func, 1, self.cpoints)

    @cache
    def ccentroid(self):
        return calculate_centroid(self.cpoints)

    @cache
    def pcentroid(self):
        return calculate_centroid(self.ppoints)

    @cache
    def max_vertex_index(self):
        return np.argmax(self.cvalues)

    @cache
    def max_vertex(self):
        return self.cpoints[self.max_vertex_index]

    @cache
    def max_value(self):
        return self.func(self.max_vertex)

    @cache
    def min_vertex_value(self):
        return np.argmin(self.cvalues)

    @cache
    def min_vertex(self):
        return self.cpoints[self.min_vertex_value]

    @cache
    def min_value(self):
        return self.func(self.min_vertex)

    @cache
    def reflection(self):
        """ get reflected version of triangle """
        return self.ccentroid + (self.ccentroid - self.max_vertex) * self.alpha

    @cache
    def reflection_value(self):
        return self.func(self.reflection)

    @cache
    def expansion(self):
        """ get the expansion point """
        return self.ccentroid + (self.reflection - self.ccentroid) * self.gamma

    @cache
    def expansion_value(self):
        return self.func(self.expansion)

    @cache
    def contraction(self):
        """ get contraction point """
        return self.ccentroid + (self.max_vertex - self.ccentroid) * self.rho

    @cache
    def contraction_value(self):
        return self.func(self.contraction)

    @cache
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
        ar[int(np.argmax(self.cvalues)), :] = new_point
        assert np.all(ar[np.argmax(self.cvalues)] == new_point)
        return ar

    def _update(self, move):
        assert move in {'reflection', 'expansion', 'contraction', 'shrink'}
        self.ppoints = self.cpoints
        self.last_move = move
        new_val = getattr(self, move)
        if move == 'shrink':
            self.cpoints = new_val
        else:
            self.cpoints = self.update_highest_point(new_val)
        self.p_min_values = self.min_value

    def iterate(self):
        """ run an iteration of simplex """
        assert self.func, 'function must be defined in order to run simplex'
        # clear cache
        self.cache = {}
        # run algorithm
        assert self.reflection not in self.cpoints
        if self.min_value < self.reflection_value < self.max_value:
            self._update('reflection')
        elif self.reflection_value < self.min_value:
            assert self.expansion not in self.cpoints
            if self.expansion_value < self.reflection_value:
                self._update('expansion')
            else:
                self._update('reflection')
        else:
            assert self.reflection_value > self.max_value
            assert self.contraction_value not in self.cpoints
            if self.contraction_value < self.max_value:
                self._update('contraction')
            else:
                self._update('shrink')


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


class PlotPlex3D:
    xvals = np.linspace(-2, 2, 100)
    yvals = np.linspace(-2, 2, 100)

    def __init__(self, simplex):
        self.simplex = simplex
        self.func = simplex.func
        assert self.func is not None, 'simplex must have function to optimize'
        self.mesh = np.stack(np.meshgrid(self.xvals, self.yvals))
        self.func_values = np.apply_along_axis(self.func, 0, self.mesh)
        self.min_val = self.func_values.min()
        self.min_val_ind = np.where(self.func_values == self.min_val)
        self.min_x = self.xvals[self.min_val_ind[0]]
        self.min_y = self.yvals[self.min_val_ind[1]]

    def plot_func(self, ax=None):
        """ plot the objective function """
        ax = ax or Axes3D(plt.gcf())
        ax.plot_surface(self.mesh[0], self.mesh[1], self.func_values,
                        alpha=0.6)
        ax.scatter([self.min_x], [self.min_y], [self.min_val], color='k')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.title('Reflection')
        return ax

    def plot_simplex(self, ax=None, verticies=None):
        ax = ax or Axes3D(plt.gcf())
        points = self.simplex.cpoints if verticies is None else verticies
        values = np.apply_along_axis(self.func, 1, points)
        # draw lines between points
        lpoints = np.vstack([points, points[0]])
        lvalues = np.append(values, values[0])
        ax.plot(lpoints[:, 0], lpoints[:, 1], lvalues)
        # draw scatter points on verticies
        ax.scatter(points[:, 0], points[:, 1], values, color='r')

    def plot_optimization(self, num_frames=8, num_iter=20):
        count = 0
        for _ in range(num_iter):
            # run one round of optimization
            make_path('Simplex')
            self.simplex.iterate()
            sequence = frame_by_frame(self.simplex.ppoints, self.simplex.cpoints,
                                      num=num_frames)
            for num, seq in enumerate(sequence):
                plt.show()
                f = plt.Figure(figsize=(9, 9))
                f = plt.gcf()  #Figure()
                ax = f.add_subplot(111, projection='3d')
                ax.view_init(elev=50)
                self.plot_func(ax=ax)
                self.plot_simplex(ax, verticies=seq)
                plt.title(self.simplex.last_move.capitalize())
                path = os.path.join('Simplex', f'{count:03d}.png')
                ax.view_init(elev=50)
                plt.savefig(path)
                count += 1
                # plot pause
                if num == len(sequence) - 1:
                    for _ in range(4):
                        path = os.path.join('Simplex', f'{count:03d}.png')
                        ax.set_title('')
                        ax.view_init(elev=50)
                        plt.savefig(path)
                        count += 1



# ----------------------------- Run animations
make2d = False
make3d = True


if make2d:  # make 2d stuff
    pp = PlotPlex2D()
    pp.plot_2d_reflection()
    pp.plot_2d_expansion()
    pp.plot_2d_contraction()
    pp.plot_2d_shrink()


if make3d:  # make 3d stuff
    points = np.array([[-2, -2], [-1.95, -1.75], [-1.75, -1.95]])
    simplex = Simplex(points, func=func)
    pp = PlotPlex3D(simplex)
    pp.plot_optimization()
    # f = plt.gcf()
    # ax = Axes3D(f)
    # ax = pp.plot_func(ax=ax)
    # pp.plot_simplex(ax=ax)
    # plt.show()


















