#!/usr/bin/python -tt
import numpy as np

from dtcwt.numpy.common import Pyramid

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
        self.include_scale = self.get_val('includescale',True)
        self.nlevels =  self.get_val('nlevels',True)
        self.biort = self.get_val('biort',False)
        self.qshift = self.get_val('qshift',False)
        self.ext_mode = max(self.get_val('ext_mode',True),4)
        self.discard_level_1 = self.get_val('discard_level_1',True)
        if self.open_cl:
            from dtcwt.opencl import Transform2d
            Transform1d = None #this has nae been implemented yet
            Transform3d = None #this has nae been implemented yet
        else:    
            from dtcwt.numpy import Transform1d,Transform2d,Transform3d
        self.transforms = [Transform1d,Transform2d,Transform3d]
        self.transform = None
        
    def __mul__(self,multiplicand):
        """
        Overloading the * operator. multiplicand is{
        forward: a numpy array
        adjoint: a wavelet transform object (WS).}
        """        
        if not self.lgc_adjoint:
            int_dimension = multiplicand.ndim
            if self.transform == None:
                if int_dimension == 3:
                    self.transform = self.transforms[2](biort=self.biort, qshift=self.qshift, \
                                                        ext_mode=self.ext_mode, \
                                                        discard_level_1=self.discard_level_1)
                else:
                    self.transform = self.transforms[int_dimension-1](biort = self.biort, qshift = self.qshift)
            td_signal = self.transform.forward(multiplicand, self.nlevels, self.include_scale)
            multiplicand = ws.WS(td_signal.lowpass,td_signal.highpasses,td_signal.scales)
            del td_signal
        else:#adjoint, multiplicand should be a WS object
            multiplicand = self.transform.inverse(Pyramid(multiplicand.ary_lowpass,multiplicand.tup_coeffs))
        return super(DTCWT,self).__mul__(multiplicand)

    class Factory:
        def create(self,ps_parameters,str_section):
            return DTCWT(ps_parameters,str_section)
