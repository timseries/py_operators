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
        self.operator_names = self.get_val('operators',False)
        self.ls_operators = [sf.create_section(ps_parameters,self.operator_names[i]) \
          for i in arange(len(self.operator_names))] #build the operator objects
        self.str_eval = self.get_mult_eval(False)  
        self.str_eval_adjoint = self.get_mult_eval(True)    
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

    def get_mult_eval(self, lgc_adjoint):
        if lgc_adjoint:
            self.ls_operators.reverse()
            str_adjoint = '~'
        str_eval = ''.join([str_adjoint + 'self.ls_operators[' + str(i) + '] * ' for i in \
                             arange(len(self.ls_operators))]) + 'multiplicand'
        if lgc_adjoint: #reverse it back to the way it was
            self.ls_operators.reverse()
        return str_eval
