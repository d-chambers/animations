# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:15:14 2015

@author: ISTI_EW
"""

import chabo as cb
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import itertools
import matplotlib.cm as cm

if __name__ == '__main__':
    ## Test options
    cb2d = 1  # Checkerboard 2d test


    ## Functions
    def checkerit(dx, xnum, v1, v2):
        """
        function to create an N dim checkerboard
        
        Parameters
        ----------
        dx : list or tuple
            List of each checker pattern dimensions (in blocks), any number of dimensions accepted
        xnum : list or tuple
            Total number of blocks in the grid in each dimension
        v1,v2 : float or int
            The velocities of each checker board block
        
        Examples
        --------
        if dx=[10,10] and xnum=[100,100] then a 2d grid with dimensions of 100 by 100 would be ceated.
        Every 10 blocks (according to dx) the velocity will be varied
        """
        if len(dx) != len(xnum):
            raise Exception('Len of dx and xnum must be the same, defines the rank of the grid')
        ar = np.ones(xnum) * v1  # initialize array with background velocity
        eye = [[1 if x == y else 0 for x in range(len(dx))] for y in range(len(dx))]
        for irow in eye:
            vpairs = [
                zip(range(0, xnum[num1] + dx[num1], dx[num1])[:-1], range(0, xnum[num1] + dx[num1], dx[num1])[1:])[
                irow[num1]::2] for num1 in range(len(dx))]  # velocity pairs
            vpairpos = itertools.product(*vpairs)
            for vpair in vpairpos:
                sli = [slice(x[0], x[1]) for x in vpair]
                ar[sli] = v2
        return ar


    def plotContours2D(cha, rayPlotStr=['r', 'b'], cmap='rainbow', savefig='2dcontour.png'):
        """
        Make a 2d plot of contours and calcualted ray paths
        
        Parameters
        ----------
        rayPlotStr : str
            A plot string recognizable by matplot lib for plotting ray paths
        """

        if len(cha.tts) < 1:
            raise Exception('No travel times calculated, run fastMarch')

        for sli, tt in cha.tts.items():
            gs = gridspec.GridSpec(2, 2, width_ratios=[10, .5], height_ratios=[10, .5], wspace=.05, hspace=.005)
            ax = plt.subplot(gs[0])
            cmesh = ax.pcolormesh(cha.xls[0], cha.xls[1], cha.velMod.transpose(), cmap=cmap, vmin=1500, vmax=4000)

            cs = ax.contour(cha.X[0], cha.X[1], tt)
            ax.plot(cha.stations.Xap, cha.stations.Yap, 'v', lw=500)
            ax.plot(cha.events.Xap, cha.events.Yap, 'o', lw=500)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel('Depth')
            ax.set_xlabel('Length')
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()
            ax1 = plt.subplot(gs[1])
            cb = plt.colorbar(cs, cax=ax1, label='Travel Time (s)')

            ax2 = plt.subplot(gs[2])
            plt.colorbar(cmesh, cax=ax2, orientation='horizontal', label='Velocity (m/s)')

            # for num, row in cha.rays.iterrows():
            #     ax.plot(np.array(row.turningPoints)[:, 0], np.array(row.turningPoints)[:, 1], '--' + rayPlotStr[num],
            #             lw=2)
            for line in cb.lines:  # change width of each line
                line.set_linewidth(20)
            ax.set_aspect('equal')
            plt.savefig(savefig, dpi=600)
            plt.show()


    # def diverge_map(high=(0.565, 0.392, 0.173), low=(0.094, 0.310, 0.635)):
    #     '''
    #     low and high are colors that will be used for the two
    #     ends of the spectrum. they can be either color strings
    #     or rgb color tuples
    #     '''
    #     c = mcolors.ColorConverter().to_rgb
    #     if isinstance(low, basestring): low = c(low)
    #     if isinstance(high, basestring): high = c(high)
    #     return make_colormap([low, c('white'), 0.5, c('white'), high])


    if cb2d:
        vdx = [200, 200]
        dx = [1, 1]
        xnum = [3000, 1800]
        xmin = [0, 0]
        vmod = np.ones(xnum)

        #        vmod[:, 0:93] = 4.6 * 1000
        #        vmod[:, 93:143] = 3.9 * 1000
        #        vmod[:, 143:206] = 3.18 * 1000
        #        vmod[:, 206:216] = 3.01 * 1000
        #        vmod[:, 216:300] = 2.73 * 1000
        vmod[:, :560] = 3.18 * 1000
        vmod[:, 560:660] = 3.01 * 1000
        vmod[:, 660:] = 2.73 * 1000
        # add LVZ
        xd = [500, 1500]
        vmod[xd[0]:xd[1], 500:560] = np.mean(vmod[:, :56]) * .708
        vmod[xd[0]:xd[1], 560:660] = np.mean(vmod[56:, :66]) * .708
        vmod[xd[0]:xd[1], 660:1200] = np.mean(vmod[66:, :]) * .708

        stations = pd.DataFrame([[1, 0, 1790, 0], [1, 2990, 1790, 0]], columns=['ID', 'X', 'Y', 'Z'])  # ID, x, y, z
        events = pd.DataFrame([[1, 50, 1500, 600, 0, 0]], columns=['ID', 'O', 'X', 'Y', 'Z', 'EType'])
        # checker = checkerit(vdx,xnum,v1,v2)


        cha = cb.Chabo(xmin=xmin, dx=dx, xnum=xnum, velMod=vmod, stations=stations, events=events)
        cha.fast_march()
        kern = cha.make_kernel()
        for num, row in cha.stations.iterrows():
            tfm = cha.tts[1.0][int(row.Xin)][int(row.Yin)]  # Travel time fast march
            tbc = np.sum((np.divide(kern[num], np.ravel(cha.kernVel))))  # Travel time back calculated
            print('fm travel time = %f, total travel time along ray path = %f' % (tfm, tbc))

        # g2 = c.from_list('grey2', [['.3', '.8']])
        g2 = cm.gray
        plotContours2D(cha, cmap=g2)
