#!/usr/bin/python -tt
from py_utils import section as s

class Operator(s.Section):
    """
    Base class for defining custom operators
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for Operator.
        """       
        super(Operator,self).__init__(ps_parameters, str_section)
        self.ws_WS = None
        self.lgc_adjoint=0
        
    def __invert__(self):
        """
        Overloading the ~ operator to indicate this object is in adjoint mode. 
        Clears the adjoint flag once a multiply happens
        """       
        self.lgc_adjoint=1

    def __mul__(self,multiplicand):
        """
        Overloading the * operator.
        """       
        if self.lgc_adjoint==1:
            self.lgc_adjoint=0
        return multiplicand
        