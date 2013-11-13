#!/usr/bin/python -tt
import dtcwt
import py_utils.signal_utilities
class Dtcwt(Operator):
    """
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for DTCWT
        """
        super(Operator,self).__init__(ps_parameters,str_section)
        
    def __mul__(self,multiplicand):
        """
        Overloading the * operator. multiplicand is:
        forward: a numpy array
        inverse: a wavelet transform object (WS).
        """
        if not self.lgc_adjoint:
            ary_size = size(mutiplicand)
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
                                                     self.dict_section['qshift'] \
                                                     self.dict_section['discard_level_1'])
                multiplcand = WS(ary_scaling,tup_coeffs)                   
        else:
            ary_size = mutiplicand.ary_size
            int_dimension = mutiplicand.int_dimension
            ary_scaling = multiplicand.ary_scaling
            tup_coeffs = multiplicand.tup_coeffs
            if int_dimension==1:
                multiplicand = dtwaveifm(ary_scaling,tup_coeffs, \
                                         self.dict_section['biort'],self.dict_section['qshift'])
            elif int_dimension==2:
                multiplicand = dtwaveifm2(ary_scaling,tup_coeffs, \
                                          self.dict_section['biort'], \
                                          self.dict_section['qshift'])
            else:
                multiplicand = dtwaveifm3(ary_scaling,tup_coeffs, \
                                          self.dict_section['biort'], \
                                          self.dict_section['qshift'], \
                                          discard_level_1=False)
        return  super(Operator,self).__mul__(multiplicand)
