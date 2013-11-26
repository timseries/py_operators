#!/usr/bin/python -tt
import numpy as np
from numpy.fft import fftn, ifftn
from scipy.ndimage.filters import convolve1d,convolve
from scipy.ndimage.filters import gaussian_filter1d,gaussian_filter
from scipy.ndimage.filters import uniform_filter1d,uniform_filter
from py_operators.operator import Operator
from py_utils.signal_utilities.sig_utils import nd_impulse, circshift
class Blur(Operator):
    """
    Operator which performs a blur in either the spatial or fourier domain.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for DTCWT
        """
        super(Blur,self).__init__(ps_parameters,str_section)
        self.str_type = self.get_val('type',False)
        self.ary_size = self.get_val('size',True)
        self.gaussian_sigma = self.get_val('gaussiansigma',True)
        self.lgc_no_circ_shift = self.get_val('lgc_no_circ_shift',True)
        self.lgc_even_fft = self.get_val('lgc_even_fft',True)
        self.int_dimension = len(self.ary_size)
        self.ary_blur_kernel = self.create_blur_kernel(self.ary_size)
        self.ary_blur_kernel_f = None
        self.spatial = self.get_val('spatial',True)
    def __mul__(self,ary_multiplicand):
        """
        Overloading the * operator. ary_multiplicand is:
        forward: a numpy array
        inverse: a wavelet transform object (WS).
        """
        if not self.spatial:
            if self.ary_blur_kernel_f==None or \
              ary_multiplicand.shape!=self.ary_blur_kernel_f.shape:
                #set up the fourier version of blur kernel once...
                blur_kernel_f = np.zeros(ary_multiplicand.shape)
                corner_indices=tuple([list(np.arange(self.ary_size[i])) for i \
                                      in np.arange(self.int_dimension)])
                blur_kernel_f[eval('np.ix_' + str(corner_indices))] = blur_kernel
                if not self.lgc_no_circ_shift:
                    blur_kernel_f = circshift(blurh_kernel_f,tuple(-self.ary_size/2))
                self.ary_blur_kernel_f = fftn(blur_kernel_f)
            if not self.lgc_adjoint:
                ary_multiplicand = ifftn(self.ary_blur_kernel_f * fftn(ary_multiplicand))
            else:
                ary_multiplicand = ifftn(np.conj(self.ary_blur_kernel_f) * fftn(ary_multiplicand))
        return super(Blur,self).__mul__(ary_multiplicand)

    def create_blur_kernel(self):
        ary_kernel = np.zeros(self.ary_size)
        if self.str_type=='uniform':
            ary_kernel[:] = 1/np.prod(self.ary_size)
        elif self.str_type=='gaussian':
            ary_impulse = nd_impulse(self.ary_size)
            gaussian_filter(ary_impulse,self.flt_gaussian_sigma,0,output=ary_kernel)
        elif self.str_type=='hamming':
            ary_kernel = np.hamming(self.ary_size[0])
        else:
            raise Exception("no such kernel " + self.str_type + " supported")    

    def get_spectrum(self):
        return self.ary_blur_kernel_f    
        
    class Factory:
        def create(self,ps_parameters,str_section):
            return Blur(ps_parameters,str_section)
