#!/usr/bin/env python3
#
#-----------------------------------------------------------------------
# SAMPLE IMPLEMENTATION: INPAINTING
#-----------------------------------------------------------------------
#
import numpy as np
import pnm
from scipy import fft
import paco
import patches
import sys
import os

#-----------------------------------------------------------------------

class PacoDeblocking(paco.PACO):
    """
    PACO JPEG deblocking

    The idea is to obtain the image whose DCT coefficients are most likely on *all* patches
    given that the DCT coefficients of those patches on the 8x8 JPEG grid fall within the
    JPEG DCT quantization bins.

    We include two terms: a weighted L1 term on the DCT coefficients which measures the
    parsimony, and an L2 fidelity term w.r.t. the input image to avoid the trivial solution
    where the DCT coefficients are all as small as they can be (given the constraints)

    The cost function (without consensus) for a given patch y is thus:

    f(x) = (1/2lambda)||y-x||_2^2 + ||D^{-1}y||_{w1}

    The constraint set is the intersection of: a) the consensus set C and b) the constraints
    on the DCT coefficients. Now, this is tricky because there are no constraints for the patches
    that like in-between the JPEG 8x8 blocks. On those patches that coincide with the JPEG blocks,
    we impose that the DCT coefficients fall in the same quantization bin as the observed quantized
    coefficients.
    So, if y is a JPEG block in the original image and a given DCT coefficient of y in the original image has a value of 'c',
    and the quantization bins for that coefficient as given by the quantization matrix are 'l' and 'u',
    we have a constraint l <= c' <= u on the corresponding coefficient in the recovered patch y'.

    The annoying problem here is that D in JPEG is not orthogonal (it's a type direct II DCT)
    """
    def __init__(self,input_signal,qtable,patch_stride):
        super().__init__(input_signal,(8,8),patch_stride)
        self.mapper        = patches.PatchMapper2D(input_signal.shape,(8,8),patch_stride)
        if self.mapper.padded_shape != self.mapper.signal_shape:
            pad_size = [ (0, a-b) for (a,b) in zip(self.mapper.padded_shape,self.mapper.signal_shape)]
            self.input_signal = np.pad(input_signal, pad_size)
        else:
            self.input_signal = input_signal
        self.qtable = qtable
        #
        # build constraints for DCT blocks
        #
        self.constraints = list()
        for k, gk in enumerate(np.ndindex(*self.grid_shape)):  # iterate over patch grid
            i,j = patch_stride * np.array(gk)
            if (i % 8 == 0) and (j % 8 == 0):
                z = fft.dctn(np.reshape(patches[k,:],(8,8)), type=2)
                zs = np.sign(z)
                zv = np.abs(z)
                zlo = np.floor(zv / self.qtable)*zv
                zhi = np.ceil(zv / self.qtable)*zv
                self.constraints.append((zlo,zhi))
            else:
                self.constraints.append(None)


    def prox_f(self, x, tau, px = None):
        """
        Proximal operator of  f(x): z = arg min f(z) + 1/2tau||z-x||^2_2

        In this case f(x) is already of that form : f(z) = f'(z) + 1/2lambda||z-y||_2^2

        The solution is another proximal operator: f(z)' + 1/2eta||z-(lambda*x+tau*y)/(lambda+tau)||_2^2
        where
        eta = tau*lambda/(tau+lambda)

        however, f'(z) is actually quite complicated:
            f'(z) = \sum w_i |(Dz)_i|

        where D is the DCT type II (non-orthogonal) transform and w_i are
        non-negative weights

        if D was orthogonal, the above solution is very easy.
        Fortunately, DD-1 is diagonal and so we can solve this by rescaling Dz,
        which is equivalent to rescaling the w_i's
        """
        if px is None:
            px = np.empty(self.mapper.patch_matrix_shape)
        np.copyto(px,x)


    def prox_g(self, x, tau, px=None):
        """
        Proximal operator of g(x): z = arg min g(x) + tau/2||z-x||^2_2
        In this case g(x) is  the indicator function of the intersection between:
        - the constraint set C
        - the box constraints B on the DCT coefficients of the DCT blocks
        The solution needs to be obtained using an interative method such as Dykstra
        (perhaps plain alterante projection works)
        """
        if px is None:
            px = np.empty(self.mapper.patch_matrix_shape)

        # constraint on C
        for J in range(10): # brutal, bad, lazy
            self.mapper.stitch(x, self.aux_signal)
            return self.mapper.extract(self.aux_signal,px)
            # constraint on box B
            for i in range(self.mapper.num_patches):
                if self.constraints[i] is not None:
                    zlo,zhi = self.constraints[i]
                    z = fft.dctn(np.reshape(px[i,:],patch_shape,type=2))
                    z[z < zlo] = zlo[z < zlo]
                    z[z > zhi] = zhi[z > zhi]
                    px[i,:] = fft.idctn(z,type=2).ravel()
        return px

    def monitor(self):
        print(np.min(self.aux_signal),np.max(self.aux_signal))
        out = np.minimum(255,np.maximum(0,self.aux_signal)).astype(np.uint8)
        pnm.imsave(f'inpainting_iter_{self.iter:04d}.pnm', out)
        pass

if __name__ == '__main__':
    dct_stats = np.loadtxt('data/dct_stats.txt')
    qtable    = np.loadtxt('data/qtable.txt')
    image     = sys.argv[1]
    ref  = pnm.imread(os.path.join('data/',image))
    jpg  = pnm.imread(os.path.join('data/jpeg/',image))
    patch_shape = (8,8)
    patch_stride = (2,2)
    paco = PacoDeblocking(jpg, qtable, patch_stride)
    paco.init()
    out = paco.run(tau=0.5,check_every=5)
    pnm.imsave('inpainting_output_.pnm',np.maximum(0,np.minimum(255,out)).astype(np.uint8))
