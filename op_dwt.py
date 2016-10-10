#!/usr/bin/python -tt
import numpy as np
import pywt
from pywt import wavedec2

from py_utils.signal_utilities import ws as ws
from py_operators.operator_ind import Operator


class DWT(Operator):
    """
    Operator which performs the forward/inverse(~) DWT, which inherits methods from
    dtcwt package.
    Returns a WS object (forward), or a numpy array (inverse)
    """

    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for DWT
        """
        super(DWT,self).__init__(ps_parameters,str_section)
        self.include_scale = self.get_val('includescale',True)
        self.nlevels =  self.get_val('nlevels',True)
        self.wavelet = self.get_val('type',False)
        if self.wavelet not in pywt.wavelist():
            raise ValueError('wavelet type ' + self.wavelet ' unrecognized')
        self.biort = self.get_val('biort',False)
        self.qshift = self.get_val('qshift',False)
        self.ext_mode = self.get_val('ext_mode',False,'zpd')
        self.discard_level_1 = self.get_val('discard_level_1',True)
        self.transforms = [wavedec,wavedec2]
        self.inverse_transforms = [wavedec,waverec2]
        self.transform = None
        self.inverse_transform = None
        self.stacker = None

    def __mul__(self,multiplicand):
        """
        Overloading the * operator. multiplicand is{
        forward: a numpy array
        adjoint: a wavelet transform object (WS).}
        """
        if not self.lgc_adjoint:
            int_dimension = multiplicand.ndim
            if self.transform == None:
                if int_dimension == 1:
                    self.transform = self.transforms[0]
                    self.inverse_transform = self.inverse_transforms[0]
                    self.stacker = np.vstack
                elif int_dimension == 2:
                    self.transform = self.transforms[1]
                    self.inverse_transform = self.inverse_transforms[1]
                    self.stacker = np.dstack
                elif int_dimension == 3:
                    raise ValueError('3d DWT not yet supported')
            coeffs = self.transform(multiplicand, wavelet=self.wavelet, level=self.nlevels, mode=self.ext_mode)
            tup_coeffs = tuple([self.stacker(coeffs[ix_]) for ix_ in xrange(len(coeffs)-1,0,-1)])
            multiplicand = ws.WS(coeffs[0],tup_coeffs,None)
            del coeffs
        else:#adjoint, multiplicand should be a WS object
            tup_coeffs = multiplicand.tup_coeffs
            coeffs_inv = [multiplicand.ary_lowpass]
            for ix in xrange(len(tup_coeffs)-1,-1,-1):
                coeffs_inv.append(tuple([tup_coeffs[ix][...,subband] for subband in xrange(tup_coeffs[ix].shape[-1])]))
            multiplicand = self.inverse_transform(coeffs_inv, wavelet=self.wavelet, mode=self.ext_mode)
        return super(DWT,self).__mul__(multiplicand)

    class Factory:
        def create(self,ps_parameters,str_section):
            return DWT(ps_parameters,str_section)
