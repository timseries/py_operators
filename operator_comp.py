#!/usr/bin/python -tt
from numpy import arange
from py_utils.section import Section
from py_utils.section_factory import SectionFactory as sf 
from py_operators.operator import Operator

class OperatorComp(Operator):
    """
    Class for defining a composition of operators
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for Operator.
        """       
        super(OperatorComp,self).__init__(ps_parameters,str_section)
        self.ls_operator_names = self.get_val('operators',False)
        self.ls_operators = [sf.create_section(ps_parameters,self.ls_operator_names[i]) \
          for i in arange(len(self.ls_operator_names))] #build the operator objects
        self.str_eval = self.get_mult_eval(False)  
        self.str_eval_adjoint = self.get_mult_eval(True)    
        self.str_eval_f = self.get_mult_eval(False,'.get_spectrum()')  
        self.str_eval_adjoint_f = self.get_mult_eval(True,'.get_spectrum()')    

    def __mul__(self,multiplicand):
        """
        Overloading the * operator. Since this relies on evaluation of a string expression, 
        don't change the input argument from 'multiplicand'
        """
        if self.lgc_adjoint:
            multiplicand = eval(self.str_eval_adjoint)
        else:
            multiplicand = eval(self.str_eval)
        return super(OperatorComp,self).__mul__(multiplicand)        

    def get_mult_eval(self, lgc_adjoint, method=''):
        """
        Returns a string used to evaluate the operator composition multiplication.
        """
        if lgc_adjoint:
            self.ls_operators.reverse()
            str_adjoint = '~'
        str_eval = ''.join([str_adjoint + 'self.ls_operators[' + str(i) + ']' + method + \
                            ' * ' for i in arange(len(self.ls_operators))])
        if method == '': #assume we want to multiply
            str_eval += 'multiplicand'               
        if lgc_adjoint: #reverse it back to the way it was
            self.ls_operators.reverse()
        return str_eval
    
    def get_spectrum(self):
        """
        Get the spectrum of the composite (by element-wise multiplying their respective spectra)
        """
        if len(self.ls_operators)==1:
            val = ls_operators[0].get_spectrum()
        else:
            val = eval(self.str_eval_f)
        return val            