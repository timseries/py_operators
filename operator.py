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
        self.lgc_adjoint = 0
        
    def __invert__(self):
        """
        Overloading the ~ operator to indicate this object is in adjoint mode. 
        Clears the adjoint flag once a multiply happens
        """       
        self.lgc_adjoint = 1
        return self
    
    def __mul__(self,multiplicand):
        """
        Overloading the * operator.
        """       
        if self.lgc_adjoint==1:
            self.lgc_adjoint = 0
        return multiplicand

    def get_spectrum(self):
        return 1