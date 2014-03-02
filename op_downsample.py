#!/usr/bin/python -tt
import numpy as np
from numpy import array, zeros, conj
from numpy import arange as ar

from py_operators.operator import Operator
from py_utils.section_factory import SectionFactory as sf

class Downsample(Operator):
    """
    Operator which performs a sampling (either upsampling or downsampling)
    using sparse matrices of logical indexes.

    Attributes:
      ds_factor (ndarray): an 1d array, each element is the sampling factor.
      for_mcand_sz (ndarray): size of the original input signal...used to determine
      correct upsampled size.
    """

    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for Downsample. 
        """
        super(Downsample,self).__init__(ps_parameters,str_section)
        self.ds_factor = self.get_val('downsamplefactor',True)
        self.offset = self.get_val('offset',True)
        if self.offset.__class__.__name__ != 'ndarray':
            self.offset = np.zeros(self.ds_factor.size,)
        self.for_mcand_sz = None
        self.tup_slices=tuple([slice(self.offset[i],None,self.ds_factor[i]) 
                                     for i in xrange(self.ds_factor.size)])
    def __mul__(self,ary_mcand):
        """
        Check superclass.
        """
        if not self.lgc_adjoint:
            #set up the downsampling slices
            if self.for_mcand_sz == None:
                self.for_mcand_sz = ary_mcand.shape
            ary_mcand = ary_mcand[self.tup_slices]
        else: #adjoint
            ary_mcand_temp = zeros(self.for_mcand_sz)
            # print ary_mcand_temp.shape
            ary_mcand_temp[self.tup_slices] = ary_mcand
            ary_mcand = ary_mcand_temp
        return super(Downsample,self).__mul__(ary_mcand)

    def get_spectrum(self):
        """Return the spectrum.
        """
        if self.lgc_adjoint:
            return 1
        else:
            return 1

    def get_spectrum_sq(self):
        """Return the squared magnitude of the spectrum.
        """
        return 1
    
    class Factory:
        def create(self,ps_parameters,str_section):
            return Downsample(ps_parameters,str_section)
