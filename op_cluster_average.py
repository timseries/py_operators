#!/usr/bin/python -tt
import numpy as np
from numpy import array, zeros, conj
from numpy import arange as ar
from scipy.sparse import csr_matrix

from py_utils.signal_utilities.ws import WS
from py_operators.operator import Operator
from py_utils.section_factory import SectionFactory as sf

class ClusterAverage(Operator):
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
        super(ClusterAverage,self).__init__(ps_parameters,str_section)
        self.grouptype = self.get_val('grouptype',False)
        self.dimension = self.get_val('dimension',False)
        self.ary_size = None
        self.A=None
        
    def __mul__(self,ls_ws_mcand):
        """
        Check superclass.
        """
        #input is a vector
        if not self.A: #need to construct the cluster summing mtx
            ws_mcand=ls_ws_mcand[0]
            ws_mcand.flatten()
            self.ary_size=ws_mcand.N
            self.dimension = ws_mcand.int_dimension
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
            ws_D_temp=WS(np.zeros(ws_mcand.ary_lowpass.shape),
                         (ws_mcand.one_subband(0)).tup_coeffs)
            #coarse to fine assignment of cluster size inverses  (lambda_Gk)
            for s in xrange(ws_D_temp.int_subbands-1,0,-1):
                if self.grouptype=='parentchildren':
                    if (s>ws_D_temp.int_subbands-ws_D_temp.int_orientations) 
                        or (s<ws_D_temp.int_orientations):
                        ws_D_temp.set_subband(s,1)
                    else:    
                        ws_D_temp.set_subband(s,1.0/self.duplicates)
                if self.grouptype=='parentchild':
                    if (s>ws_D_temp.int_subbands-ws_D_temp.int_orientations):
                        ws_D_temp.set_subband(s,1.0/(self.duplicates-1))
                    elif (s<ws_D_temp.int_orientations):
                        ws_D_temp.set_subband(s,1.0)
                    else:    
                        ws_D_temp.set_subband(s,1.0/self.duplicates)
            total_size=ws_D_temp.N
            lowpass_size=ws_D_temp.ary_lowpass.size
            csr_cols=np.tile(np.arange(0,total_size*self.duplicates),self.duplicates)
            csr_rows=np.tile(np.arange(0,ws_D_temp.N),self.duplicates)
            csr_data_template=ws_D_temp.flatten()
            csr_data=np.tile(csr_data_template,self.duplicates)
            self.A=csr_matrix((csr_data,(csr_rows,csr_cols)),shape=(total_size,total_size*self.duplicates))
        if not self.lgc_adjoint:    
            #preallocate a vectore to store the ws_mcand in
            vec_ix=0
            ary_xhat=np.zeros(self.ary_size*self.duplicates,)
            #we need to flatten a list of ws objects
            for ws_mcand in ls_ws_mcand:
                ary_xhat[vec_ix:vec_ix+self.ary_size]=ws_mcand.flatten()
                vec_ix+=self.ary_size
            #now apply the matrix/vector product
            ary_xhat=np.dot(self.A,ary_xhat)
            
            #return a single ws object (x_bar)
            ws_D_temp=WS(np.zeros(ws_mcand.ary_lowpass.shape),
                         (ws_mcand.one_subband(0)).tup_coeffs)
            ws_D_temp.ws_vector=ary_xhat
            ws_D_temp.unflatten()
            return ws_D_temp
        else: 
            #flatten a single ws object (should only be one ws obj in the list)
            ary_x=ls_ws_mcand[0].flatten()
            #apply the matrix
            ary_x=np.dot(self.A.transpose(),ary_x)
            #create a list of ws objects
            vec_ix=0
            ls_ws_result=[]
            for j in xrange(self.duplicates):
                ws_D_temp=WS(np.zeros(ws_mcand.ary_lowpass.shape),
                             (ws_mcand.one_subband(0)).tup_coeffs)
                ws_D_temp.ws_vector=ary_x[vec_ix:vec_ix+self.ary_size]
                vec_ix+=self.ary_size
                ls_ws_result.append(ws_D_temp)
            return ls_ws_result
        return super(ClusterAverage,self).__mul__(ary_mcand)

    class Factory:
        def create(self,ps_parameters,str_section):
            return ClusterAverage(ps_parameters,str_section)
