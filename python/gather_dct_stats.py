#!/usr/bin/env python3
#
#-----------------------------------------------------------------------
# SAMPLE IMPLEMENTATION: INPAINTING
#-----------------------------------------------------------------------
#
import os
import numpy as np
import pnm
from scipy import fft
import paco
import patches


if __name__ == '__main__':
    patch_shape = (8,8)
    patch_stride = (1,1)
    stats = np.zeros((8,8))
    n = 0
    with open('data/images.txt') as list_file:
        for entry in list_file:
            print(entry.strip(),end=' ')
            #
            # read image
            #
            img = pnm.imread(os.path.join('data',entry.strip()))
            #
            # extract patches
            #
            M,N   = img.shape
            gm,gn = patches.grid_size(M,8,1),patches.grid_size(N,8,1)
            n += gm*gn
            for i in range(gm):
                if i % 100 == 0:
                    print('.', end='',flush=True)
                for j in range(gn):
                    # transform
                    z = fft.dctn(img[i:i+8,j:j+8],type=2)
                    # gather  stats
                    stats += np.abs(z)
            print()
            print(np.round(stats/n))
    np.savetxt('data/dct_stats.txt',np.round(stats/n),fmt='%8d')