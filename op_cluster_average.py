#!/usr/bin/python -tt
import numpy as np
from numpy import array, zeros, conj
from numpy import arange as ar
from scipy.sparse import csr_matrix

from py_utils.signal_utilities.ws import WS
from py_operators.operator import Operator
from py_operators.op_average import Average

class ClusterAverage(Average):
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
        self.average_type='cluster'
        
    def __mul__(self,ls_ws_mcand):
        """
        Check superclass.
        input is a list of ws objects
        """
        #if pkl-saved version of A doesn't exist, create one and save    
        
        if not self.initialized:
            self.init_csr_avg(ls_ws_mcand[0])
        if not self.lgc_adjoint:    
            if self.duplicates!=len(ls_ws_mcand):
                raise ValueError('there are not enough WS-es'+
                                 ' for the replicated variable space')
            #preallocate a vector to store the ws_mcand in
            vec_ix=0
            ary_xhat=np.zeros(self.int_size*self.duplicates,)
            #we need to flatten a list of ws objects
            for ws_mcand in ls_ws_mcand:
                ary_xhat[vec_ix:vec_ix+self.int_size]=ws_mcand.flatten()
                vec_ix+=self.int_size
            #now apply the matrix/vector product
            ary_xhat=self.csr_avg*ary_xhat
            #return a single ws object (x_bar)
            ws_csr_avg_data=WS(np.zeros(ws_mcand.ary_lowpass.shape),
                         (ws_mcand.one_subband(0)).tup_coeffs)
            ws_csr_avg_data.ws_vector=ary_xhat
            ws_csr_avg_data.unflatten()
            ary_mcand=ws_csr_avg_data
        else: 
            #flatten a single ws object (should only be one ws obj in the list)
            if ls_ws_mcand.__class__.__name__=='list':
                ws_mcand=ls_ws_mcand[0]
            else:    
                ws_mcand=ls_ws_mcand
            ary_x=ws_mcand.flatten()
            #apply the matrix
            ary_x=self.csr_avg.transpose()*ary_x
            #create a list of ws objects
            vec_ix=0
            ls_ws_result=[]
            for int_dup in xrange(self.duplicates):
                ws_csr_avg_data=WS(np.zeros(ws_mcand.ary_lowpass.shape),
                             (ws_mcand.one_subband(0)).tup_coeffs)
                ws_csr_avg_data.unflatten(ary_x[vec_ix:vec_ix+self.int_size])
                vec_ix+=self.int_size
                ls_ws_result.append(ws_csr_avg_data)
            ary_mcand=ls_ws_result
        return super(ClusterAverage,self).__mul__(ary_mcand)

    def create_csr_avg(self,ws_mcand):
        ws_mcand.flatten()
        #preallocate the row locations
        ls_ws_csr_avg_data=[WS(np.zeros(ws_mcand.ary_lowpass.shape),
                     (ws_mcand.one_subband(0)).tup_coeffs)
                     for j in xrange(self.duplicates)]
        #coarse to fine assignment of cluster size inverses (lambda_Gk)
        for int_dup in xrange(self.duplicates):
            ws_csr_avg_data=ls_ws_csr_avg_data[int_dup]
            for s in xrange(self.S-1,0,-1):
                if self.grouptype=='parentchildren':
                    if ((s>=self.S-self.theta) or (s<=self.theta)):
                        if int_dup==0: #the singleton copy for non-overlapped variables
                            ws_csr_avg_data.set_subband(s,1.0)
                    else:    
                        ws_csr_avg_data.set_subband(s,1.0/self.duplicates)

                if self.grouptype=='parentchild':
                    if (s>=self.S-self.theta):
                        if int_dup>0:
                            ws_csr_avg_data.set_subband(s,1.0/(self.duplicates-1))
                    elif (s<=self.theta):
                        if int_dup==0:
                            ws_csr_avg_data.set_subband(s,1.0)
                    else:    
                        ws_csr_avg_data.set_subband(s,1.0/self.duplicates)

        csr_cols=np.arange(0,self.int_size*self.duplicates)
        csr_rows=np.tile(np.arange(0,self.int_size),self.duplicates)
        csr_data=np.zeros(self.int_size*self.duplicates)
        vec_ix=0
        for ws_csr_avg_data in ls_ws_csr_avg_data:
            csr_data[vec_ix:vec_ix+self.int_size]=ws_csr_avg_data.flatten()
            vec_ix+=self.int_size
        self.csr_avg=csr_matrix((csr_data,(csr_rows,csr_cols)),
                          shape=(self.int_size,self.int_size*self.duplicates))

    class Factory:
        def create(self,ps_parameters,str_section):
            return ClusterAverage(ps_parameters,str_section)
