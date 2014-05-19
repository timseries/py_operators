#!/usr/bin/python -tt
import numpy as np
from numpy import array, zeros, conj
from numpy import arange as ar
from scipy.sparse import csr_matrix

from py_utils.signal_utilities.ws import WS
from py_operators.operator import Operator
from py_utils.section_factory import SectionFactory as sf

class ClusterSum(Operator):
    """
    Operator which performs a sampling (either upsampling or downsampling)
    using sparse matrices of logical indexes.

    Attributes:
      ds_factor (ndarray): an 1d array, each element is the sampling factor.
      for_mcand_sz (ndarray): size of the original input signal...used to determine
      correct upsampled size.
    """

    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for Downsample. 
        """
        super(ClusterSum,self).__init__(ps_parameters,str_section)
        self.grouptype = self.get_val('grouptype',False)
        self.dimension = self.get_val('dimension',False)
        self.ary_size = None
        self.D=None
        
    def __mul__(self,ls_ws_mcand):
        """
        Check superclass.
        """
        #input is a vector
        if not self.D: #need to construct the cluster summing mtx
            ws_mcand=ls_ws_mcand[0]
            ws_mcand.flatten()
            if self.grouptype=='parentchildren':
                    self.duplicates=2
                elif self.grouptype=='parentchild':    
                    self.duplicates=2**self.dimension+1
                else:
                    raise ValueError('unsupported grouptype')
                #populate the cluster structure
                if self.duplicates!=len(ls_ws_mcand):
                    raise ValueError('there are not enough WS for the replicated variable space')
                #preallocate the row locations
                D_indices=np.zeros()
                ws_D_temp=WS(np.zeros(ws_mcand.ary_lowpass.shape),
                             (ws_mcand.one_subband(0)).tup_coeffs)
                #coarse to fine assignment of cluster size inverses  (lambda_Gk)
                for s in xrange(ws_D_temp.int_subbands-1,0,-1):
                    if self.grouptype=='parentchildren':
                        if (s>ws_D_temp.int_subbands-ws_D_temp.int_orientations):
                            ws_D_temp.set_subband(s,0)
                        elif (s<ws_D_temp.int_orientations):
                            ws_D_temp.set_subband(s,0)
                        else:    
                            ws_D_temp.set_subband(s,1.0/self.duplicates)
                    if self.grouptype=='parentchild':
                        if (s>ws_D_temp.int_subbands-ws_D_temp.int_orientations):
                            ws_D_temp.set_subband(s,1.0/(self.duplicates-1))
                        elif (s<ws_D_temp.int_orientations):
                            ws_D_temp.set_subband(s,0)
                        else:    
                            ws_D_temp.set_subband(s,1.0/self.duplicates)
                        
                total_size=ws_D_temp.N
                lowpass_size=ws_D_temp.ary_lowpass.size
                csr_cols=np.tile(np.arange(0,total_size*self.duplicates),self.duplicates)
                csr_rows=np.tile(np.arange(0,ws_D_temp.N),self.duplicates)
                # csr_data_template=np.hstack((np.zeros(lowpass_size,),np.ones(total_size-lowpass_size,)))
                csr_data_template=ws_D_temp.flatten()
                csr_data=np.tile(csr_data_template,self.duplicates)
                self.D=csr_matrix((csr_data,(csr_rows,csr_cols)),shape=(total_size,total_size*self.duplicates))
        if not self.lgc_adjoint:    
            if self.for_mcand_sz == None:
                self.for_mcand_sz = ary_mcand.shape
            ary_mcand = ary_mcand[self.tup_slices]
        else: #adjoint
            ary_mcand_temp = zeros(self.for_mcand_sz)
            # print ary_mcand_temp.shape
            ary_mcand_temp[self.tup_slices] = ary_mcand
            ary_mcand = ary_mcand_temp
        return super(ClusterSum,self).__mul__(ary_mcand)

    class Factory:
        def create(self,ps_parameters,str_section):
            return ClusterSum(ps_parameters,str_section)
