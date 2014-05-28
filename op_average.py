#!/usr/bin/python -tt
import numpy as np
from numpy import array, zeros, conj
from numpy import arange as ar
from scipy.sparse import csr_matrix

from py_utils.signal_utilities.ws import WS
from py_operators.operator import Operator
from py_utils.section_factory import SectionFactory as sf

class Average(Operator):
    """
    Operator which performs the group averaging for each element of the 
    replicated variable space.

    Attributes:
      correct upsampled size.
    """

    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for Downsample. 
        """
        super(Average,self).__init__(ps_parameters,str_section)
        self.grouptype=self.get_val('grouptype',False)
        self.dimension=None
        self.ary_size=None
        self.duplicates=None
        self.A=None
        self.S=None
        self.L=None
        self.theta=None
        self.average_type=None
        
    def init_A_params(self,ls_ws_mcand):
        """
        Check superclass.
        """
        ws_mcand=ls_ws_mcand[0]
        ws_mcand.flatten()
        self.int_size=ws_mcand.N
        self.tup_shape=ws_mcand.ary_shape
        self.L=ws_mcand.int_levels
        self.dimension=ws_mcand.int_dimension
        self.S=ws_mcand.int_subbands
        self.theta=ws_mcand.int_orientations
        self.sparse_matrix_in=self.get_val('sparsematrixinput',False)
        if self.grouptype=='parentchildren':
            self.duplicates=2
        elif self.grouptype=='parentchild':    
            self.duplicates=2**self.dimension+1
        else:
            raise ValueError('unsupported grouptype')
            #populate the cluster structure
        if self.duplicates!=len(ls_ws_mcand):
            raise ValueError('there are not enough WS-es
                              for the replicated variable space')
        self.file_string=(self.average_type+'_'+
                         self.grouptype+'_'+
                         str(self.L)+'_'+
                         str(self.theta)+'_'+
                         str(self.tup_shape)+'.pkl')

    def load_A(self):
        sec_input=sf.create_section(self.get_params(),self.sparse_matrix_in)
        sec_input.filepath+=self.file_string
            self.A=sec_input.read({},True)
        return self.A
    
    def load_A(self):
        sec_input=sf.create_section(self.get_params(),self.sparse_matrix_in)
        sec_input.filepath+=self.file_string
            self.A=sec_input.read({},True)
        return self.A

    class Factory:
        def create(self,ps_parameters,str_section):
            return Average(ps_parameters,str_section)
