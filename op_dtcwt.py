#!/usr/bin/python -tt
import numpy as np
from dtcwt import dtwavexfm, dtwaveifm
from dtcwt import dtwavexfm2, dtwaveifm2
from dtcwt import dtwavexfm3, dtwaveifm3, biort, qshift
from py_utils.signal_utilities import ws as ws
from py_operators.operator import Operator
class DTCWT(Operator):
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
        self.nlevels =  self.get_val('nlevels',True)
        self.biort = self.get_val('biort',False)
        self.qshift = self.get_val('qshift',False)
        self.ext_mode = max(self.get_val('ext_mode',True),4)
        self.discard_level_1 = self.get_val('discard_level_1',True)
        
    def __mul__(self,multiplicand):
        """
        Overloading the * operator. multiplicand is{
        forward: a numpy array
        adjoint: a wavelet transform object (WS).}
        """
        if not self.lgc_adjoint:
            int_dimension = multiplicand.ndim
            if int_dimension==1:
                ary_scaling,tup_coeffs = dtwavexfm(multiplicand, \
                                                     self.nlevels, \
                                                     self.biort, \
                                                     self.qshift)
            elif int_dimension==2:
                ary_scaling,tup_coeffs = dtwavexfm2(multiplicand, \
                                                     self.nlevels, \
                                                     self.biort, \
                                                     self.qshift)
            else:
                ary_scaling,tup_coeffs = dtwavexfm3(multiplicand, \
                                                     self.nlevels, \
                                                     self.biort, \
                                                     self.qshift, \
                                                     self.ext_mode, \
                                                     self.discard_level_1)
            multiplicand = ws.WS(ary_scaling,tup_coeffs)
        else:#adjoint
            int_dimension = multiplicand.int_dimension
            ary_scaling = multiplicand.ary_scaling
            tup_coeffs = multiplicand.tup_coeffs
            if int_dimension==1:
                multiplicand = dtwaveifm(ary_scaling,tup_coeffs, \
                                            self.biort, \
                                            self.qshift)
            elif int_dimension==2:
                multiplicand = dtwaveifm2(ary_scaling,tup_coeffs, \
                                             self.biort, \
                                             self.qshift)
            else:
                multiplicand = dtwaveifm3(ary_scaling,tup_coeffs, \
                                             self.biort, \
                                             self.qshift, \
                                             self.ext_mode)
        return super(DTCWT,self).__mul__(multiplicand)

    class Factory:
        def create(self,ps_parameters,str_section):
            return DTCWT(ps_parameters,str_section)
