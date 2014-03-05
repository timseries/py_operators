#!/usr/bin/python -tt
from py_utils.section import Section

class Operator(Section):
    """
    Base class for defining custom operators
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for Operator.
        """       
        super(Operator,self).__init__(ps_parameters,str_section)
        self.output_fourier = 0
        self.lgc_adjoint = 0
        
    def __invert__(self):
        """
        Overloading the ~ operator to indicate this object is in adjoint mode. 
        Clears the adjoint flag once a multiply happens
        """       
        self.lgc_adjoint = 1
        return self
    
    def t(self):
        """
        Also performs the operator adjoint/Hermetian transpose.
        """             
        self.__invert__()
        return self

    def set_output_fourier(self,val):
        """
        Force the output to be fourier samples
        """
        self.output_fourier = val
    
    def __mul__(self,multiplicand):
        """
        Overloading the * operator.
        """       
        self.lgc_adjoint = 0
        return multiplicand

    def get_spectrum(self):
        return 1