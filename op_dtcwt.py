#!/usr/bin/python -tt
import numpy as np
from dtcwt import *
from py_utils.signal_utilities import ws as ws
from py_operators import operator as op
class DTCWT(op.Operator):
    """
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for DTCWT
        """
        super(DTCWT,self).__init__(ps_parameters,str_section)
        
    def __mul__(self,multiplicand):
        """
        Overloading the * operator. multiplicand is:
        forward: a numpy array
        inverse: a wavelet transform object (WS).
        """
        if not self.lgc_adjoint:
            int_dimension = multiplicand.ndim
            if int_dimension==1:
                ary_scaling,tup_coeffs = dtwavexfm(multiplicand, \
                                                     self.dict_section['nlevels'], \
                                                     self.dict_section['biort'], \
                                                     self.dict_section['qshift'])
            elif int_dimension==2:
                ary_scaling,tup_coeffs = dtwavexfm2(multiplicand, \
                                                       self.dict_section['nlevels'], \
                                                       self.dict_section['biort'], \
                                                       self.dict_section['qshift'])
            else:
                ary_scaling,tup_coeffs = dtwavexfm3(multiplicand, \
                                                       self.dict_section['nlevels'], \
                                                       self.dict_section['biort'], \
                                                       self.dict_section['qshift'], \
                                                       self.dict_section['discard_level_1'])
                multiplcand = ws.WS(ary_scaling,tup_coeffs)                   
        else:
            int_dimension = multiplicand.int_dimension
            ary_scaling = multiplicand.ary_scaling
            tup_coeffs = multiplicand.tup_coeffs
            if int_dimension==1:
                multiplicand = dtwaveifm(ary_scaling,tup_coeffs, \
                                            self.dict_section['biort'], \
                                            self.dict_section['qshift'])
            elif int_dimension==2:
                multiplicand = dtwaveifm2(ary_scaling,tup_coeffs, \
                                             self.dict_section['biort'], \
                                             self.dict_section['qshift'])
            else:
                multiplicand = dtwaveifm3(ary_scaling,tup_coeffs, \
                                             self.dict_section['biort'], \
                                             self.dict_section['qshift'], \
                                             self.dict_section['discard_level_1'])
        return super(DTCWT,self).__mul__(multiplicand)

    class Factory:
        def create(self,ps_parameters,str_section):
            return DTCWT(ps_parameters,str_section)
