#!/usr/bin/python -tt
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from numpy import conj, sqrt

from py_operators.operator import Operator
from py_utils.section_factory import SectionFactory as sf
from py_utils.signal_utilities.sig_utils import pad_center, crop_center

import pdb

class SampledFT(Operator):
    """
    Operator which performs a sampled FT. 

    
    Attributes:
        mask (bool ndarray): The optional binary mask which specifies which samples of the DFT matrix
            to use in the forward/inverse DFT.
    
    """
    
    def __init__(self,ps_parameters,str_section):
        """Class constructor for SampledFT. 
        """
        super(SampledFT,self).__init__(ps_parameters,str_section)
        self.output_fourier = 1
        #setup the mask
        mask_sec_in = self.get_val('masksectioninput',False)
        self.mask = None
        if mask_sec_in:
            sec_mask = sf.create_section(self.get_params(), mask_sec_in)
            self.mask = sec_mask.read({}, True)
        
    def __mul__(self,ary_mcand):
        """Perform forward or adjoint sampled FFT. If no mask is present
        simply perform a a fully sampled forward or adjoint FFT.
        
        """
        if not self.lgc_adjoint:
            ary_mcand = ary_mcand.real
            ary_mcand = 1/sqrt(ary_mcand.size)*fftshift(fftn(ifftshift(ary_mcand)))
            if self.mask is not None and (ary_mcand.shape != self.mask.shape):
                ary_mcand = crop_center(ary_mcand, self.mask.shape)
            if self.mask is not None:    
                ary_mcand *= self.mask
        else:    
            ary_mcand = ary_mcand.real
            if self.mask is not None and (ary_mcand.shape != self.mask.shape):
                ary_mcand = pad_center(ary_mcand,self.mask.shape)
            ary_mcand = sqrt(ary_mcand.size)*ifftshift(ifftn(fftshift(ary_mcand)))
            
        return super(SampledFT,self).__mul__(ary_mcand)
        
    class Factory:
        def create(self,ps_parameters,str_section):
            return SampledFT(ps_parameters,str_section)
