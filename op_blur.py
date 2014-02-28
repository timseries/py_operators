#!/usr/bin/python -tt
import numpy as np
from numpy.fft import fftn, ifftn
from numpy import conj, array
from numpy import arange as ar
from numpy import minimum as min, maximum as max
from scipy.ndimage.filters import convolve1d,convolve
from scipy.ndimage.filters import gaussian_filter1d,gaussian_filter
from scipy.ndimage.filters import uniform_filter1d,uniform_filter
from py_operators.operator import Operator
from py_utils.section_factory import SectionFactory as sf
from py_utils.signal_utilities.sig_utils import nd_impulse, circshift, colonvec, gaussian

import pdb

class Blur(Operator):
    """
    Operator which performs a blur in either the spatial or Fourier 
    domain. The lgc_even_fftn option is a python port for the blurring
    operator which does not assume periodic boundary conditions. 
    In other words, this option performs an implicit convolution.

    Note: 
      Lgc_even_fft is used to support convolution in which there 
      is no periodic boundary assumption. Therefore, this option
      does a padding of the input array, does the fourier domain
      convolution, then crops the invalid border data from the result.
    
    Attributes:
      str_type (str): uniform, cylindrical, gaussian, pyramid
      spatial (int): 1 to do a convolution with spatial filters.
      output_fourier (int): 1 to ouput Fourier coefficients.
      gaussian_sigma (float): stdev of gaussian blur.
      lgc_even_fft (itn): 1 to use even-length FFT and pad psf.
      ary_sz (ndarray): the spatial support of the kernel.
      kernel (ndarray): the kernel 'image'.
      int_dimension (int): the dimension of the operator.
      for_kernel_f (ndarray): the Fourier forward blur.
      for_sz_max (ndarray): the max of kernel and input dimensions.
      size_min (ndarray): the min of kernel and input dimensions.
      fft_sz (ndarray): the size argument of fftn.
      for_mcand_sz (ndarray): multiplcand dimensions forward blur.
      adj_mcand_sz (ndarray): multiplcand dimensions adjoint blur.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for Blur. 
        """
        super(Blur,self).__init__(ps_parameters,str_section)
        self.str_type = self.get_val('type',False)
        self.spatial = self.get_val('spatial',True)
        self.output_fourier = self.get_val('outputfourier',True)
        self.gaussian_sigma = self.get_val('gaussiansigma',True)
        self.lgc_even_fft = self.get_val('evenfft',True)
        self.ary_sz = None
        self.create_kernel()
        self.int_dimension = len(self.ary_sz)
        self.for_kernel_f = None
        self.for_sz_max = None
        self.size_min = None
        self.fft_sz = None
        self.for_mcand_sz = None
        self.adj_mcand_sz = None
        
    def __mul__(self,ary_mcand):
        """
        Check superclass.
        """
                                                                                
        if self.spatial:
            ValueError('spatial domain blurring not supported')
        if not self.lgc_adjoint:
            #set up the Fourier blur kernel once for this mcand size
            if ary_mcand.shape != self.for_mcand_sz:
                self.for_mcand_sz = ary_mcand.shape
                if self.lgc_even_fft:
                    self.for_sz_max = max(self.kernel.shape,ary_mcand.shape)
                    self.for_sz_min = min(self.kernel.shape,ary_mcand.shape)
                    self.for_fft_sz = (self.for_sz_max + 
                                         np.mod(self.for_sz_max,2))
                    self.for_kernel_f = self.kernel
                else:
                    self.for_kernel_f = np.zeros(ary_mcand.shape)
                    self.for_sz_max = self.for_kernel_f.shape
                    self.for_sz_min = self.for_kernel_f.shape
                    self.for_fft_sz = self.for_kernel_f.shape
                    c_i = tuple([list(ar(self.ary_sz[i])) 
                                 for i in ar(self.int_dimension)])
                    #embed and circularly shift by half the support
                    c_i = eval('np.ix_' + str(c_i))
                    self.for_kernel_f[c_i] = self.kernel
                    self.for_kernel_f = circshift(self.for_kernel_f,
                                                  tuple((-self.ary_sz/2.0).astype('uint16')))
                    #take the fft, with the correct size
                self.for_kernel_f = fftn(self.for_kernel_f,
                                         s=self.for_fft_sz)
            ary_mcand = self.for_kernel_f * fftn(ary_mcand,
                                                 s=self.for_fft_sz)
            if not self.output_fourier and not self.lgc_even_fft:
                ary_mcand = np.real(ifftn(ary_mcand))
            if self.lgc_even_fft: #we must unpad first
                ary_mcand = np.real(ifftn(ary_mcand)) 
                ary_mcand = ary_mcand[colonvec(self.for_sz_min,
                                               self.for_sz_max)]
                if self.output_fourier:
                    ary_mcand = fftn(ary_mcand)
        else: #adjoint
            if ary_mcand.shape != self.adj_mcand_sz:
                self.adj_mcand_sz = ary_mcand.shape #L
                if self.lgc_even_fft:
                    self.adj_sz_min = min(array(self.adj_mcand_sz)+
                                            array(self.kernel.shape)-1,
                                            2*array(self.adj_mcand_sz)
                                            -1)
                    self.adj_fft_sz = (self.adj_sz_min + 
                                         np.mod(self.adj_sz_min,2))
                    self.adj_kernel_f = conj(fftn(self.kernel,s=self.adj_fft_sz))
                else:
                    self.adj_sz_min = self.for_kernel_f.shape
                    self.adj_fft_sz = self.for_kernel_f.shape
                    self.adj_kernel_f = conj(self.for_kernel_f)
            if self.lgc_even_fft: #pad first
                ary_result_temp = np.zeros(self.adj_sz_min)
                unpad_slices = colonvec(array(self.adj_sz_min)-
                                        array(self.adj_mcand_sz)+1,
                                        array(self.adj_sz_min))
                ary_result_temp[unpad_slices] = ary_mcand
                ary_mcand = ary_result_temp
            ary_mcand = self.adj_kernel_f * fftn(ary_mcand,s=self.adj_fft_sz)
            if not self.output_fourier and not self.lgc_even_fft:
                ary_mcand = np.real(ifftn(ary_mcand))
            if self.lgc_even_fft: #unpad in spatial domain
                ary_mcand = np.real(ifftn(ary_mcand))
                unpad_slices = colonvec(np.ones(self.int_dimension,),
                                        array(self.adj_sz_min)-
                                        max(array(self.kernel.shape)-
                                            array(self.adj_mcand_sz),0))
                ary_mcand = ary_mcand[unpad_slices]
                if self.output_fourier:
                    ary_mcand = fftn(ary_mcand)
                        
        return super(Blur,self).__mul__(ary_mcand)

    def create_kernel(self):
        """Creates or reads from file the kernel weights and stores
        in the class kernel attribute.
        """
        if self.str_type!='file':
            self.ary_sz = self.get_val('size',True)
            ary_kernel = np.zeros(self.ary_sz)
        if self.str_type =='uniform':
            ary_kernel[:] = 1.0 / np.prod(self.ary_sz)
        elif self.str_type =='gaussian':
            ary_impulse = nd_impulse(self.ary_sz)
            gaussian_filter(ary_impulse,self.gaussian_sigma,
                            0,output=ary_kernel)
            ary_kernel = gaussian(self.ary_sz,self.gaussian_sigma)
        elif self.str_type =='hamming':
            ary_kernel = np.hamming(self.ary_sz[0])
        elif self.str_type=='file':    
            sec_input = sf.create_section(self.ps_parameters,
                                          self.get_val('filesection',False))
            ary_kernel = sec_input.read({},True)
            self.ary_sz = array(ary_kernel.shape)
        else:
            ValueError("no such kernel "+self.str_type+" supported")
        self.kernel=ary_kernel

    def get_spectrum(self):
        """Return the spectrum.
        """
        if self.lgc_adjoint:
            return self.adj_kernel_f
        else:
            return self.for_kernel_f    

    def get_spectrum_sq(self):
        """Return the squared magnitude of the spectrum.
        """
        return conj(self.for_kernel_f)*self.for_kernel_f
    
    class Factory:
        def create(self,ps_parameters,str_section):
            return Blur(ps_parameters,str_section)
