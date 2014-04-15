#!/usr/bin/env python

import numpy as np
from scipy.io import loadmat
from numpy import savez_compressed
from util import summarise_mat, summarise_cube
import pdb

verif_temp = loadmat('./matlab/verification.mat')
verif = dict((k,v) for k, v in verif_temp.iteritems() if (not k.startswith('_') and not k.startswith('qbgn')))
verif_cube = dict((k,v) for k, v in verif_temp.iteritems() if (not k.startswith('_') and k.startswith('qbgn')))
del verif_temp

for idx, v in enumerate(verif['lena_blur_2D']):
    verif['lena_blur_2D_{0}'.format(idx)] = v[0]
del verif['lena_blur_2D']

for idx, v in enumerate(verif['cell_blur_3D']):
    verif['cell_blur_3D_{0}'.format(idx)] = v[0]
del verif['cell_blur_3D']    

summaries = dict((k, v) for k, v in verif.iteritems())
#for summarizable matrices
for k,v in verif_cube.iteritems():
    summaries[k] = v
    
savez_compressed('verification.npz', **summaries)

# Convert qbgn.mat -> qbgn.npz
savez_compressed('qbgn.npz', **loadmat('../dtcwt/tests/qbgn.mat'))
