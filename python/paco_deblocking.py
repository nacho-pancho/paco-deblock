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
import sys
import argparse


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

    Since the DCT used by JPEG is orthogonal, the distances between patches in the proximal operators are
    preserved in the coefficient space. This allows us to work in DCT space for most of the operations
    (with the exception of extracting/stitching)

    """
    def __init__(self,input_signal,patch_stride,qtable,lam,weights,ref):
        super().__init__(input_signal,(8,8),patch_stride)
        #
        # problem-specific parameters
        #
        self.lam = lam
        self.weights = weights.ravel()
        self.qtable = qtable.ravel()
        self.ref = ref
        #
        # build constraints for DCT blocks
        #
        print("extracting patches")
        self.input_patches = self.mapper.extract(self.input_signal)
        print("constructing constraint set")
        print("computing input DCT coefficients")
        self.input_coeffs = np.empty(self.input_patches.shape)
        self.do_dct(self.input_patches,self.input_coeffs)
        self.box_hi =  10000*np.ones(self.mapper.patch_matrix_shape)
        self.box_lo = -10000*np.ones(self.mapper.patch_matrix_shape)
        for k, gk in enumerate(np.ndindex(*self.mapper.grid_shape)):  # iterate over patch grid
            i,j = patch_stride * np.array(gk)
            if (i % 8 == 0) and (j % 8 == 0):
                z = fft.dctn(np.reshape(self.input_patches[k,:],(8,8)),norm='ortho', type=2).ravel()
                zlo = np.floor(z / self.qtable)*self.qtable
                zhi = zlo + self.qtable
                self.box_lo[k,:] = zlo
                self.box_hi[k,:] = zhi

    @staticmethod
    def do_dct(x,xdct):
        for k in range(x.shape[0]):
            xdct[k,:] = fft.dctn(np.reshape(x[k, :], (8, 8)), norm='ortho', type=2).ravel()


    @staticmethod
    def do_idct(xdct,x):
        for k in range(x.shape[0]):
            x[k,:] = fft.idctn(np.reshape(xdct[k, :], (8, 8)), norm='ortho', type=2).ravel()

    def prox_f(self, x, tau, px = None):
        """
        Proximal operator of  f(x): z = arg min f(z) + 1/2tau||z-x||^2_2

        In this case f(z) is already of that form : f(z) = f'(z) + 1/2lambda||z-y||_2^2

        The solution is another proximal operator: f(z)' + 1/2eta||z-(lambda*x+tau*y)/(lambda+tau)||_2^2
        where
        eta = tau*lambda/(tau+lambda)

        in our case,
            f'(z) = \sum w_i |z_i|

        where w_i are non-negative weights
        """
        print('prox_f')
        if px is None:
            px = np.empty(self.mapper.patch_matrix_shape)
        np.copyto(px,x)

        for k in range(self.mapper.num_patches):
            x0k = self.input_coeffs[k,:] #
            xk  = x[k,:]
            aux = (self.lam*xk + tau*x0k)*(1.0/(self.lam+tau))
            # weighted soft thresholding
            eta = (self.lam*tau/(self.lam+tau))*self.weights
            aaux = np.abs(aux)
            saux = np.sign(aux)
            aaux -= eta
            aaux[aaux < 0] = 0
            px[k,:] = saux * aaux

    def init(self):
        """
        Problem-specific initialization
        """
        self.A[:] = 0
        self.A[:] = 0
        self.B[:] = np.copy(self.input_coeffs)
        self.U[:] = 0
        self.prevB[:] = np.copy(self.input_coeffs)
        self.iter = 0


    def prox_g(self, x, tau, px=None):
        """
        Proximal operator of g(x): z = arg min g(x) + tau/2||z-x||^2_2
        In this case g(x) is  the indicator function of the intersection between:
        - the constraint set C
        - the box constraints B on the DCT coefficients of the DCT blocks

        The solution needs to be obtained using an interative method such as Dykstra
        (perhaps plain alterante projection works)

        The input x is actually in DCT space, so we need to convert to and from signal
        space when doing the stitching/extraction
        """
        print('prox_g')
        if px is None:
            px = np.empty(self.mapper.patch_matrix_shape)
        np.copyto(px,x)
        for J in range(10): # brutal, bad, lazy
            prevx = np.copy(px)
            self.do_idct(px,px)
            # constraint on C
            self.mapper.stitch(px, self.aux_signal)
            self.mapper.extract(self.aux_signal, px)
            self.do_dct(px,px)
            # constraint on box B
            viol_hi = np.mean(np.maximum(0,px-self.box_hi))
            viol_lo = np.mean(np.maximum(0,self.box_lo-px))
            px = np.minimum(self.box_hi,np.maximum(self.box_lo,px))
            print('POCS',J,end=' ')
            dif = np.linalg.norm(px-prevx,'fro')/(1e-10+np.linalg.norm(px,'fro')),
            print('dif',dif,end=' ')
            print('viol: lo',viol_lo,'up',viol_hi)
            if dif < 1e-3:
                break
        print()
        return px

    def prox_g_dykstra(self, x, tau, px=None):
        """
        Proximal operator of g(x): z = arg min g(x) + tau/2||z-x||^2_2
        In this case g(x) is  the indicator function of the intersection between:
        - the constraint set C
        - the box constraints B on the DCT coefficients of the DCT blocks

        The input x is actually in DCT space, so we need to convert to and from signal
        space when doing the stitching/extraction

        This version uses Dykstra's alternate projection method
        """
        print('prox_g')
        if px is None:
            px = np.empty(self.mapper.patch_matrix_shape)
        np.copyto(px,x)
        p = np.zeros(px.shape)
        q = np.zeros(px.shape)
        y = np.zeros(px.shape)
        for J in range(50): # brutal, bad, lazy
            prevx = np.copy(px)
            self.do_idct(px+p,px)
            # constraint on C
            #
            # 1) y <- proj(x+p)
            #
            self.mapper.stitch(px, self.aux_signal)
            self.mapper.extract(self.aux_signal, px)
            self.do_dct(px,y)
            #
            # 2) p <- p + x - y
            #
            p += px
            p -= y
            # constraint on box B
            #
            # 3) x <- proj(y+q)
            #
            viol_hi = np.sum(np.maximum(0,px-self.box_hi))
            viol_lo = np.sum(np.maximum(0,self.box_lo-px))
            px = np.minimum(self.box_hi,np.maximum(self.box_lo,px+q))
            #
            # 4) q <- q + y - x
            #
            q += y
            q -= px
            print('POCS',J,end=' ')
            dif = np.linalg.norm(px-prevx,'fro')/(1e-10+np.linalg.norm(px,'fro')),
            print('dif',dif,end=' ')
            print('viol: lo',viol_lo,'up',viol_hi)
            if dif < 1e-3:
                break
        print()
        return px


    def monitor(self):
        #print(np.min(self.aux_signal),np.max(self.aux_signal))
        out = np.minimum(255,np.maximum(0,self.aux_signal+128)).astype(np.uint8)
        pnm.imsave(f'inpainting_iter_{self.iter:04d}.pnm', out)
        mse = np.sqrt(np.mean(np.square(self.ref - out)))
        print('cost',np.mean(np.abs(self.A)),'mse',mse)


if __name__ == '__main__':
    epilog = "Output image file name is built from input name and parameters."
    parser = argparse.ArgumentParser(epilog=epilog)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-s", "--stride", type=int, default=1,
                        help="patch stride")
    parser.add_argument("-t", "--tau", type=float, default=1.0,
                        help="ADMM stepsize")
    parser.add_argument("-l", "--lam", type=float, default=1.0,
                        help="ADMM stepsize")
    parser.add_argument("-o", "--output", type=str, default="deblocked.pgm",
                        help="output file.")
    parser.add_argument("-S", "--stats", type=str, default="data/dct_stats.txt",
                        help="DCT stats, for DCT coeff. weights.")

    parser.add_argument("input", help="input image file")
    parser.add_argument("qtable", help="mask image file")
    parser.add_argument("ref", help="dictionary file (ASCII matrix, one atom per row)")
    args = parser.parse_args()

    cmd = " ".join(sys.argv)
    print(("Command: " + cmd))

    dct_stats = np.loadtxt('data/dct_stats.txt')
    qtable    = np.loadtxt(args.qtable)
    input  = pnm.imread(args.input).astype(float)-128
    tau       = args.tau
    lam       = args.lam*128

    print('tau',tau,'lambda (after. scaling)',lam)
    ref  = pnm.imread(args.ref).astype(float)
    patch_shape = (8,8)
    patch_stride = (args.stride,args.stride) # bad and fast, for testing
    dct_stats = dct_stats/np.min(dct_stats)
    weights = 1.0/dct_stats
    paco = PacoDeblocking(input, patch_stride, qtable,lam, weights,ref)
    paco.init()
    paco.run(tau,check_every=1)
    out = paco.aux_signal+128
    pnm.imsave('deblocking_output.pnm',np.maximum(0,np.minimum(255,out)).astype(np.uint8))
