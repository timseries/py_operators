#!/usr/bin/python -tt
from py_utils import Section

class Operator(Section):
    """
    Base class for defining custom operators
    """
    
    def __init__(self,ps_parameters, str_section):
        """
        Class constructor for Operator.
        """       
        super(Section,self).__init__(ps_parameters, str_section)
        self.wsWS = None