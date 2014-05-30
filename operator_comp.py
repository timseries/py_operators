#!/usr/bin/python -tt
from numpy import arange, conj, real
from numpy.fft import fftn, ifftn
from py_utils.section import Section
from py_utils.section_factory import SectionFactory as sf
from py_operators.operator import Operator

class OperatorComp(Operator):
    """
    Class for defining a composition of operators
    """
    
    def __init__(self,ps_params,str_section):
        """
        Class constructor for Operator.
        """       
        super(OperatorComp,self).__init__(ps_params,str_section)
        self.ls_op_names = self.get_val('operators',False)
        if self.ls_op_names.__class__.__name__ == 'str':
            self.ls_op_names = [self.ls_op_names]
        #build the operator objects            
        self.ls_ops = [sf.create_section(ps_params,self.ls_op_names[i])
                       for i in arange(len(self.ls_op_names))] 
        self.str_eval = self.get_mult_eval(False)  
        self.str_eval_adjoint = self.get_mult_eval(True)    
        self.str_eval_f = self.get_mult_eval(False,'.get_spectrum()')  
        self.str_eval_adjoint_f = self.get_mult_eval(True,
                                                     '.get_spectrum()')    
        self.str_eval_psd = self.get_mult_eval(False,'.get_spectrum_sq()')  
        
    def __mul__(self,multiplicand):
        """
        Overloading the * operator. Since this relies on evaluation of a string expression, 
        don't change the input argument from 'multiplicand'
        """
        if self.lgc_adjoint:
            multiplicand = eval(self.str_eval_adjoint)
        else:
            multiplicand = eval(self.str_eval)
        if self.output_fourier:
            multiplicand = fftn(multiplicand)
        return super(OperatorComp,self).__mul__(multiplicand)        

    def get_mult_eval(self, lgc_adjoint, method=''):
        """
        Returns a string used to evaluate the operator composition multiplication.
        """
        str_adjoint = ''
        
        if lgc_adjoint:
            order_iterator = xrange(len(self.ls_ops))
            str_adjoint = '~'
        else:
            order_iterator = xrange(len(self.ls_ops)-1,-1,-1)    
        str_eval = ''.join(['(' +str_adjoint + 'self.ls_ops[' + str(i) + ']' + method + \
                            ' * ' for i in order_iterator])
        if method == '': #assume we want to multiply
            str_eval += 'multiplicand'               
        else: #remove the last ' * '
            str_eval = str_eval[:-3]    
        str_eval += ')'*len(self.ls_ops)
        return str_eval

    def get_spectrum(self):
        """
        Get the spectrum of the composite (by element-wise multiplying their respective spectra)
        """
        if len(self.ls_ops)==1:
            val = self.ls_ops[0].get_spectrum()
        else:
            val = eval(self.str_eval_f)
        return val            
    
    def get_spectrum_sq(self):
        """
        Get the spectrum of the composite (by element-wise multiplying their respective spectra)
        """
        if len(self.ls_ops)==1:
            val = self.ls_ops[0].get_psd()
        else:
            val = eval(self.str_eval_f)
            val = real(conj(val) * val)
        return val            