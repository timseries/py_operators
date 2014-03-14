#!/usr/bin/python -tt
import numpy as np

from dtcwt.numpy.common import Pyramid

from py_utils.signal_utilities import ws as ws
from py_utils.section_factory import SectionFactory as sf
from py_operators.operator import Operator

class Scattering(Operator):
    """
    Operator which performs the forward/inverse(~) Scattering transform attributed to Mallat.
    Returns a WS object (forward), or a numpy array (inverse)
    """
    
    def __init__(self,ps_params,str_section):
        """
        Class constructor for Scattering
        """
        super(Scattering,self).__init__(ps_params,str_section)
        self.depth = self.get_val('depth',False)
        self.transform_sec = self.get_val('transform',False)
        self.W = sf.create_section(ps_params,self.transform_sec)
        #create the versions of W we'll need in a list
        self.max_transform_levels = self.W.get_val('nlevels')
        self.W = []
        for j in xrange(self.max_transform_levels+1,0,-1):
            ps_params.set_key_val_pairs(self.transform_sec,['nlevels'],[j])
            self.W.append(sf.create_section(ps_params,self.transform_sec))
        
    def __mul__(self,multiplicand):
        """
        Overloading the * operator. multiplicand is{
        forward: a numpy array
        adjoint: a wavelet transform object (WS).}
        """        
        W = self.W
        if not self.lgc_adjoint:
            print 'inverse scattering not implemented yet'
        else:#adjoint, multiplicand should be a WS object
            for j in xrange(self.depth):
                W[j]*multiplicand
                
        return super(Scattering,self).__mul__(multiplicand)

    class Factory:
        def create(self,ps_parameters,str_section):
            return Scattering(ps_parameters,str_section)
