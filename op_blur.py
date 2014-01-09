#!/usr/bin/python -tt
import numpy as np
from numpy.fft import fftn, ifftn
from numpy import conj, array
from scipy.ndimage.filters import convolve1d,convolve
from scipy.ndimage.filters import gaussian_filter1d,gaussian_filter
from scipy.ndimage.filters import uniform_filter1d,uniform_filter
from py_operators.operator import Operator
from py_utils.section_factory import SectionFactory as sf
from py_utils.signal_utilities.sig_utils import nd_impulse, circshift, colonvec

class Blur(Operator):
    """
    Operator which performs a blur in either the spatial or fourier domain.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for Blur
        """
        super(Blur,self).__init__(ps_parameters,str_section)
        self.str_type = self.get_val('type',False)
        self.gaussian_sigma = self.get_val('gaussiansigma',True)
        self.lgc_even_fft = self.get_val('evenfft',True)
        self.ary_size = None
        self.blur_kernel = self.create_blur_kernel()
        self.int_dimension = len(self.ary_size)
        self.forward_blur_kernel_f = None
        self.spatial = self.get_val('spatial',True) #for spatial convolution option, later...
        self.forward_size_max = None #larger of the size of the multiplicand and psf
        self.size_min = None #smaller of the size of the multiplicand and psf
        self.fft_size = None #the size of the fft we'll use
        self.forward_multiplicand_shape = None
        self.adjoint_multiplicand_shape = None
        
    def __mul__(self,ary_multiplicand):
        """
        Overloading the * operator. ary_multiplicand is:
           forward: a numpy array, spatial domain object
           inverse: a wavelet transform object (WS).

        Returns a fourier domain object
        Lgc_even_fft is used to support convolution in which there is no periodic boundary assumption. 
        Therefore, this option does a padding of the input array, does the fourier domain convolution, then crops the invalid
        border data from the result.

        This also assumes the forward has been called for this object before adjoint is ever called.

        Currentially 'spatial' is unsupported
        
        """
        if not self.spatial:
            if not self.lgc_adjoint:
                if ary_multiplicand.shape != self.forward_multiplicand_shape:
                    #set up the fourier version of blur kernel once for this multiplicand size
                    self.forward_multiplicand_shape = ary_multiplicand.shape
                    #if even_fft, we'll pad our multiplicand and psf so they're both even length
                    if self.lgc_even_fft:
                        self.forward_size_max = np.maximum(self.blur_kernel.shape,ary_multiplicand.shape)
                        self.forward_size_min = np.minimum(self.blur_kernel.shape,ary_multiplicand.shape)
                        self.forward_fft_size = self.forward_size_max + np.mod(self.forward_size_max,2)
                        self.forward_blur_kernel_f = self.blur_kernel
                    else:
                        self.forward_blur_kernel_f = np.zeros(ary_multiplicand.shape)
                        self.forward_size_max = self.forward_blur_kernel_f.shape
                        self.forward_size_min = self.forward_blur_kernel_f.shape
                        self.forward_fft_size = self.forward_blur_kernel_f.shape
                        corner_indices = tuple([list(np.arange(self.ary_size[i])) for i \
                                          in np.arange(self.int_dimension)])
                        #embed the blur kernel in the top left, and circularly shift by half the support
                        self.forward_blur_kernel_f[eval('np.ix_' + str(corner_indices))] = self.blur_kernel
                        self.forward_blur_kernel_f = circshift(self.forward_blur_kernel_f,tuple(-self.ary_size/2))
                        #take the fft, with the correct size
                    self.forward_blur_kernel_f = fftn(self.forward_blur_kernel_f,s=self.forward_fft_size)
                ary_multiplicand = self.forward_blur_kernel_f * fftn(ary_multiplicand,s=self.forward_fft_size)
                if self.lgc_even_fft:
                    ary_multiplicand = ifftn(ary_multiplicand)
                    ary_multiplicand = ary_multiplicand[colonvec(self.forward_size_min, self.forward_size_max)]
                    ary_multiplicand = fftn(ary_multiplicand)
            else:#adjoint
                if ary_multiplicand.shape != self.adjoint_multiplicand_shape:
                    self.adjoint_multiplicand_shape = ary_multiplicand.shape
                    if self.lgc_even_fft:
                        self.adjoint_size_min = np.minimum(array(self.adjoint_multiplicand_shape) + \
                                                           array(self.blur_kernel.shape) - 1, \
                                                           2 * array(self.adjoint_multiplicand_shape) - 1)
                        self.adjoint_fft_size = self.adjoint_size_min + np.mod(self.adjoint_size_min,2)
                        self.adjoint_blur_kernel_f = conj(fftn(self.blur_kernel,s=self.adjoint_fft_size))
                    else:
                        self.adjoint_size_min = self.forward_blur_kernel_f.shape
                        self.adjoint_fft_size = self.forward_blur_kernel_f.shape
                        self.adjoint_blur_kernel_f = conj(self.forward_blur_kernel_f)
                if self.lgc_even_fft: #unpad first           
                    ary_result_temp = np.zeros(self.adjoint_size_min)
                    ary_result_temp[colonvec(array(self.adjoint_size_min) - \
                                             array(self.adjoint_multiplicand_shape) + 1,\
                                             array(self.adjoint_size_min))] = ary_multiplicand

                    ary_multiplicand = ary_result_temp
                ary_multiplicand = self.adjoint_blur_kernel_f * fftn(ary_multiplicand,s=self.adjoint_fft_size)
                if self.lgc_even_fft: #unpad in spatial domain
                    ary_multiplicand = ifftn(ary_multiplicand)
                    ary_multiplicand = ary_multiplicand[colonvec(np.ones(self.int_dimension,),\
                                                                 array(self.adjoint_size_min) - \
                                                                 np.maximum(array(self.blur_kernel.shape) - \
                                                                    array(self.adjoint_multiplicand_shape),0))]
                    ary_multiplicand = fftn(ary_multiplicand)
        else:
            raise Exception("not coded yet, use input class here")    
        return super(Blur,self).__mul__(ary_multiplicand)

    def create_blur_kernel(self):
        ary_kernel = np.zeros(self.ary_size)
        if self.str_type!='file':
            self.ary_size = self.get_val('size',True)

        if self.str_type =='uniform':
            ary_kernel[:] = 1.0 / np.prod(self.ary_size)
        elif self.str_type =='gaussian':
            ary_impulse = nd_impulse(self.ary_size)
            gaussian_filter(ary_impulse,self.flt_gaussian_sigma,0,output=ary_kernel)
        elif self.str_type =='hamming':
            ary_kernel = np.hamming(self.ary_size[0])
        elif self.str_type=='file':    
            sec_input = sf.create_section(self.ps_parameters,self.get_val('filesection',False))
            ary_kernel = sec_input.read({},True)
            self.ary_size = array(ary_kernel.shape)
        else:
            raise Exception("no such kernel " + self.str_type + " supported")    
        return ary_kernel

    def get_spectrum(self):
        return self.forward_blur_kernel_f    
        
    class Factory:
        def create(self,ps_parameters,str_section):
            return Blur(ps_parameters,str_section)
