#!/usr/bin/python -tt
import numpy as np
from numpy.fft import fftn, ifftn
from numpy import hamming
from scipy.ndimage.filters import convolve1d,convolve
from scipy.ndimage.filters import gaussian_filter1d,gaussian_filter
from scipy.ndimage.filters import uniform_filter1d,uniform_filter
from py_operators.operator import Operator
from py_utils.matrix_utils import circshift
from py_utils.signal_utilities.sig_utils import nd_impulse
class Blur(Operator):
    """
    Operator which performs a blur in either the spatial or fourier domain.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for DTCWT
        """
        super(Blur,self).__init__(ps_parameters,str_section)
        self.type = self.get_val('type',False)
        self.size = self.get_val('size',True)
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
            blur_kernel_f[eval('np.ix_' + str(corner_indices))]=blur_kernel
            blur_kernel_f = circshift(blur_kernel_f,tuple(-self.size/2))
            self.blur_kernel_f = fftn(blur_kernel_f)
            
        if not self.lgc_adjoint:
            multiplicand = ifftn(self.blur_kernel_f * fftn(multiplicand))
        else:
            multiplicand = ifftn(conj(self.blur_kernel_f) * fftn(multiplicand))
        return super(Blur,self).__mul__(multiplicand)

    def create_blur_kernel(self,size):
        ary_kernel = np.zeros(self.size)
        if self.type=='uniform':
            ary_kernel[:] = 1/np.prod(self.size)
        elif self.type=='gaussian':
            ary_impulse = nd_impulse(self.size)
            gaussian_filter(ary_impulse,self.gaussian_sigma,0,output=ary_kernel)
        elif self.type=='hamming':
            ary_kernel = np.hamming(self.size[0])
        else:
            raise Exception("no such kernel " + self.type + " supported")    
        
    class Factory:
        def create(self,ps_parameters,str_section):
            return Blur(ps_parameters,str_section)
