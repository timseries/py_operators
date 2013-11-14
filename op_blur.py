#!/usr/bin/python -tt
import numpy as np
from np.fft import fftn, ifftn
from np import hamming
from scipy.ndimage.filters import convolve1d,convolve
from scipy.ndimage.filters import gaussian_filter1d,gaussian_filter
from scipy.ndimage.filters import uniform_filter1d,uniform_filter
from py_operators.operator import Operator
from py_utils.matrix_utils import circshift
from py_utils.signal_utilities.sig_utils import nd_impulse
class Blur(Operator):
    """
    Operator which performs the forward/inverse(~) DTCWT, which inherits methods from 
    dtcwt package authored by Rich Wareham. 
    Returns a WS object (forward), or a numpy array (inverse)
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for DTCWT
        """
        super(DTCWT,self).__init__(ps_parameters,str_section)
        self.type = self.get_val('type',False)
        self.size = np.array(self.get_val('size',True))
        self.gaussian_sigma = self.get_val('gaussiansigma',True)
        self.dimension = len(self.size)
        self.blur_kernel = self.create_blur_kernel(self.size)
        self.blur_kernel_f = None
        
    def __mul__(self,multiplicand):
        """
        Overloading the * operator. multiplicand is:
        forward: a numpy array
        inverse: a wavelet transform object (WS).
        """
        if self.blur_kernel_f==None or multiplicand.shape!=self.blur_kernel_f.shape:
            #set up the fourier version of blur kernel once...
            blur_kernel_f = np.zeros(multiplicand.shape)
            corner_indices=tuple([list(np.arange(self.size[i])) for i \
                                  in np.arange(self.dimension)])
            blur_kernel_f(eval('np.ix_' + str(corner_indices)))=blur_kernel
            blur_kernel_f = circshift(blur_kernel_f,tuple(-self.size/2))
            self.blur_kernel_f = fftn(blur_kernel_f)
            
        if not self.lgc_adjoint:
            multiplicand = ifftn(self.blur_kernel_f * fftn(multiplicand))
        else:
            multiplicand = ifftn(conj(self.blur_kernel_f) * fftn(multiplicand))
        return super(Blur,self).__mul__(multiplicand)

    def create_blur_kernel(self,size)
        ary_impulse = nd_impulse(self.size*2+1)
        if self.type=='uniform'
            
    
    class Factory:
        def create(self,ps_parameters,str_section):
            return Blur(ps_parameters,str_section)
