# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 21:00:16 2015

@author: isti_ew
"""

import struct
import numpy as np

#First want to simply read in the velocity values
binname = "TestDatasets/velmod23.P.mod.buf"
headname = "TestDatasets/velmod23.P.mod.hdr"

gridxnum = [80,56,40]
gridxmax = [13280.,6330.,2750.]
griddx = [50.,50.,50.]

with open(binname,'rb') as binfile:
    binvelocs = []
    for block in iter(lambda: binfile.read(4),""):
        binvelocs.append(struct.unpack('f',block)[0])

#Read in the header and use it to size a grid for the velocity 
#Will need to either make sure that it matches the existing Chabo grid, or give the user the option to replace the Chabo grid?
with open(headname,'r') as headfile:
    header = headfile.read().split()
    
headxnum = [int(header[0])-1,int(header[1])-1,int(header[2])-1]
headdx = [float(header[6])*1000,float(header[7])*1000,float(header[8])*1000]
headxmax = [float(header[3])*1000+headxnum[0]*headdx[0],float(header[4])*1000+headxnum[1]*headdx[1],float(header[5])*-1000]

print headxnum
print headdx
print headxmax

if not (headxnum == gridxnum):
    raise TypeError ('NLL grid xnum %s does not match Chabo grid xnum %s' % (headxnum,gridxnum))
if not (headxmax == gridxmax):
    raise TypeError ('NLL grid xmax %s does not match Chabo grid xmax %s' % (headxmax,gridxmax))
if not (headdx == griddx):
    raise TypeError ('NLL grid dx %s does not match Chabo grid dx %s' % (headdx,griddx))
    
velocity = np.ones(headxnum)
i,j,l=0,0,0
k = headxnum[2]-1
x = 0
while i<headxnum[0]:
    while j<headxnum[1]:
        while k>=0:
            velocity[i,j,k]=binvelocs[l]
            print (i,j,k)
            print x
            print l
            l = l + 1
            k = k - 1
            x = x + 1
        j = j + 1
        l = l + 1
        k = headxnum[2]-1
    i = i + 1
    l = l + headxnum[2] + 1
    j = 0

#Figure out which velocity value goes to which index in the grid and insert it #How to deal with the fact that it works with nodes instead of centers?  Does it just duplicate the last (or first?) velocity in each direction?
#Values are stored in succession of planes in x, consisting of vectors in y, and points in z... therefore to unloop it go backwards and assign along vertical columns, which construct planes