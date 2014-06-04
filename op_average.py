#!/usr/bin/python -tt
import numpy as np
from numpy import array, zeros, conj
from numpy import arange as ar
import cPickle

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
        duplicates: the maximum number of duplicates for any given variable
        """
        
        super(Average,self).__init__(ps_parameters,str_section)
        self.grouptype=self.get_val('grouptype',False)
        self.duplicates=None
        self.dimension=None
        self.ary_size=None
        self.csr_avg=None
        self.S=None
        self.L=None
        self.theta=None
        self.average_type=None
        self.file_path=None
        self.initialized=False

    def init_csr_avg(self,ws_mcand):
        """
        Initializes the parameters needed to create sparse Averaging matrix (self.csr_avg)
        from a WS object. Intuitively, properties of the matrix should depend on the 
        vector it's multiplying.
        """
        if ws_mcand.__class__.__name__!='WS':
            raise ValueError('ws_mcand needs to be a WS')
        self.int_size=ws_mcand.size
        self.tup_shape=ws_mcand.ary_shape
        self.L=ws_mcand.int_levels
        self.dimension=ws_mcand.int_dimension
        if self.grouptype=='parentchildren':
            self.duplicates=2
            self.groupsize=2**self.dimension+1
        elif self.grouptype=='parentchild':    
            self.duplicates=2**self.dimension+1
            self.groupsize=2
        else:
            raise ValueError('unsupported grouptype: ' + self.grouptype)
        self.S=ws_mcand.int_subbands
        self.theta=ws_mcand.int_orientations
        self.sparse_matrix_input=self.get_val('sparsematrixinput',False)
        #populate the cluster structure
        self.file_string=(self.average_type+'_'+
                         self.grouptype+'_'+
                         str(self.L)+'_'+
                         str(self.theta)+'_'+
                         str(self.tup_shape).replace(' ','') +'.pkl')
        if not self.load_csr_avg():    
            self.create_csr_avg(ws_mcand)
            self.save_csr_avg()
        self.initialized=True

    def load_csr_avg(self):
        sec_input=sf.create_section(self.get_params(),self.sparse_matrix_input)
        sec_input.filepath+=self.file_string
        self.file_path=sec_input.filepath
        self.csr_avg=sec_input.read({},True)
        return self.csr_avg!=None

    def save_csr_avg(self):
        if not self.file_path:
            self.file_path=self.ps_parameters.str_file_dir+self.file_string
        filehandler=open(self.file_path, 'wb')
        cPickle.dump(self.csr_avg,filehandler)

    class Factory:
        def create(self,ps_parameters,str_section):
            return Average(ps_parameters,str_section)
