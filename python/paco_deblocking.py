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

    """
    def __init__(self,input_signal,input_mask,patch_stride):
        super().__init__(input_signal,(8,8),patch_stride)
        self.mapper        = patches.PatchMapper2D(input_signal.shape,(8,8),patch_stride)
        if self.mapper.padded_shape != self.mapper.signal_shape:
            pad_size = [ (0, a-b) for (a,b) in zip(self.mapper.padded_shape,self.mapper.signal_shape)]
            self.input_mask = np.pad(input_mask, pad_size)
        else:
            self.input_signal = input_mask


    def prox_f(self, x, tau, px = None):
        """
        Proximal operator of  f(x): z = arg min f(x) + tau/2||z-x||^2_2
        In PACO, f(x) is the main cost function to be minimized
        """
        if px is None:
            px = np.empty(self.mapper.patch_matrix_shape)
        #
        # sample: soft thresholding
        #
        np.copyto(px,x)
        # terribly slow:
        for i in range(self.mapper.num_patches):
            patch = np.reshape(px[i,:],patch_shape)
            fft.dctn(patch,norm="ortho",overwrite_x=True)
            patch[np.abs(patch) < tau] = 0
            fft.idctn(patch,norm="ortho",overwrite_x=True)
            px[i, :] = patch.ravel()
        return px


    def prox_g(self, x, tau, px=None):
        """
        Proximal operator of g(x): z = arg min g(x) + tau/2||z-x||^2_2
        In PACO, g(x) is usually the indicator function of the constraint set
        which again is usually the intersection between the consensus set and
        additional constraints imposed by the problem.
        
        The sample implementation below just projects onto the consensus set.
        """
        if px is None:
            px = np.empty(self.mapper.patch_matrix_shape)
    
        self.mapper.stitch(x, self.aux_signal)
        self.aux_signal[self.input_mask] = self.input_signal[self.input_mask] # inpainting constraint
        return self.mapper.extract(self.aux_signal,px)

    def monitor(self):
        print(np.min(self.aux_signal),np.max(self.aux_signal))
        out = np.minimum(255,np.maximum(0,self.aux_signal)).astype(np.uint8)
        pnm.imsave(f'inpainting_iter_{self.iter:04d}.pnm', out)
        pass

if __name__ == '__main__':
    print("PACO - DCT - INPAINTING")
    print("*** NOTE -- THIS IS SUPER SLOW ***")
    _ref_  = pnm.imread("../data/test_nacho.pgm").astype(np.double)
    _mask_ = pnm.imread("../data/test_nacho_mask.pbm").astype(bool)
    _input_= _ref_ * _mask_ # zero out data within mask
    pnm.imsave('inpainting_input_.pnm',_input_.astype(np.uint8))
    pnm.imsave('inpainting_mask_.pnm',_mask_.astype(np.uint8))
    patch_shape = (8,8)
    patch_stride = (2,2)
    paco = PacoDctInpainting(_input_,_mask_, patch_shape, patch_stride)
    paco.init()
    _output_ = paco.run(tau=0.5,check_every=5)
    pnm.imsave('inpainting_output_.pnm',np.maximum(0,np.minimum(255,_output_)).astype(np.uint8))
