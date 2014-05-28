#!/usr/bin/python -tt
import numpy as np
from numpy import array, zeros, conj
from numpy import arange as ar
from scipy.sparse import csr_matrix

from py_utils.signal_utilities.ws import WS
from py_operators.op_average import Average
from py_utils.section_factory import SectionFactory as sf

class GroupAverage(Average):
    """
    Operator which performs the group averaging for each element of the 
    replicated variable space. The ouptut vectors is the same size as the 
    input vector (the group averages are copied for each element)

    Attributes:
      correct upsampled size.
    """

    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for GroupAverage. 
        """
        super(GroupAverage,self).__init__(ps_parameters,str_section)
        self.average_type='group'
        
    def __mul__(self,ls_ws_mcand):
        """
        Check superclass.
        """
        #input is a vector
        if not self.A: #need to construct the group summing mtx
            self.init_A_params(ls_ws_mcand)
            self.load_A(ls_ws_mcand)
        if not self.load_A:    
            ws_mcand=ls_ws_mcand[0]
            ws_mcand.flatten()
            #preallocate the row locations
            ws_A_data=WS(np.zeros(ws_mcand.ary_lowpass.shape),
                         (ws_mcand.one_subband(0)).tup_coeffs)
            ls_ws_A_groups=[WS(np.zeros(ws_mcand.ary_lowpass.shape),
                               (ws_mcand.one_subband(0)).tup_coeffs) 
                               for j in xrange(self.duplicates)]
            #coarse to fine assignment of group size inverses (1/gi)
            #and assignemnt of group numbers
            int_g_count=1 #zero is reserved for absence of group,start at 1
            if self.grouptype=='parentchildren': 
                #interleave the non-overlapping group number
                #assignments in the two replicated copies
                for int_offset in xrange(self.duplicates):
                    ws_A_groups=ls_ws_A_groups[int_offset]
                    int_s_start=self.S-1-self.theta*int_offset
                    parent_indices=[ar(int_s_start-2*j*self.theta,
                                       int_s_start-2*j*self.theta-self.theta,-1) 
                                       for j in xrange(int_s_start/self.theta/2)]
                    parent_indices=np.hstack(parent_indices)
                    for s in parent_indices:
                        int_s_size=ws_A_groups.get_subband(s).size
                        tup_s_shape=ws_A_groups.get_subband(s).shape
                        group_indices=ar(int_g_count,int_s_size+1).reshape(tup_s_shape)
                        ws_A_groups.set_subband(s,group_indices)
                        ws_A_groups.set_subband(s-self.theta,ws_A_groups.get_upsampled_parent(s))
                        int_g_count+=int_s_size
            elif self.grouptype=='parentchild':
                #each entry corresponds to a level and is a dict itself
                #store for fast lookup of children indices from the parent index
                dict_level_indices={} 
                #now assign each group elementwise, slower but the code is clearer
                int_g_count=1
                for s in xrange(self.S-1,self.theta-1,-1):
                    ws_A_groups=ls_ws_A_groups[0]
                    int_level,int_orientation=ws_A_groups.lev_ori_from_subband(s)
                    int_s_size=ws_A_groups.get_subband(s).size
                    tup_s_shape=ws_A_groups.get_subband(s).shape
                    if not dict_child_indices.has_key(int_level):
                        dict_level_indices[int_level]={}
                    dict_child_indices=dict_level_indices[int_level]
                    #gather the flattened subbands, there are K replicates
                    ls_flat_parents=[ls_ws_A_groups[int_dup].get_subband(s).flatten()
                                     for int_dup in xrange(self.duplicates)]
                    ls_flat_children=[ls_ws_A_groups[int_dup].get_subband(s-self.theta).flatten()
                                      for int_dup in xrange(self.duplicates)]
                    for int_parent_ix in xrange(int_s_size):
                        int_skip=0
                        for int_child_ix in xrange(2**self.dimension):
                            if not dict_child_indices.has_key(int_child_ix):
                                #get the child indices 
                                ary_dummy_child=np.zeros(tup_s_shape*2,dtype=bool)
                                ary_dummy_child[ws_A_groups.ds_slices[int_child_ix]]=1
                                dict_child_indices[int_child_ix]=np.nonzero(ary_dummy_child.flatten())
                            if ls_flat_parents[int_child_ix][int_parent_ix]>0:
                                int_skip=1
                            ls_flat_parents[int_child_ix+int_skip][int_parent_ix]=int_g_count
                            int_ls_ws_ix=int_child_ix+1+int_skip
                            if int_ls_ws_ix>=self.duplicates:
                                int_ls_ws_ix=0 #wrap
                            ls_flat_children[int_ls_ws_ix][dict_child_indices[int_child_ix][int_parent_ix]]=int_g_count
                            int_g_count+=1
                    #store group numbers back in the ws structures, for this s
                    for int_dup in xrange(self.duplicate):
                        ls_ws_A_groups[int_dup].set_subband(s,ls_flat_parents[int_dup].reshape(tup_s_shape))
                        ls_ws_A_groups[int_dup].set_subband(s-self.theta,ls_flat_children[int_dup].reshape(2*tup_s_shape))
                del dict_child_indices
                del ls_flat_parents
                del ls_flat_children
            else:
                raise ValueError('unsupported group type')            
            #now we can build the sparse matrix rows/cols by iterating
            #through the group numbers
            #first flatten the group numbers
            ary_csr_groups=np.zeros(self.int_size*self.duplicates,dtype=uint32)
            ix_=0
            for int_dup in xrange(self.duplicates):
                ary_csr_groups[ix_:ix_+self.int_size]=ls_ws_A_groups[int_dup].flatten()
                ix_+=self.int_size
            del ls_ws_A_groups    
            #next build the group columns and rows    
            csr_rows=np.zeros(0,)
            csr_cols=np.zeros(0,)
            for int_group_index in xrange(1,int_g_count):
                g_ix_locs=np.nonzero(ary_csr_groups==int_group_index)
                if len(g_ix_locs)>0:
                    csr_cols=np.hstack((csr_cols,np.tile(g_ix_locs,len(g_ix_locs))))
                    csr_rows=np.hstack((csr_rows,np.repeat(g_ix_locs,len(g_ix_locs))))
            csr_data=1.0/self.duplicates*np.ones(csr_rows.size,)
            self.A=csr_matrix((csr_data,(csr_rows,csr_cols)),
                              shape=(self.int_size*self.duplicates,self.int_size*self.duplicates))
        if 1: #should give the same result for transpose since matrix is symmetric
            #preallocate a vector to store the ws_mcand in
            vec_ix=0
            ary_xhat=np.zeros(self.int_size*self.duplicates,)
            #we need to flatten a list of ws objects
            for ws_mcand in ls_ws_mcand:
                ary_xhat[vec_ix:vec_ix+self.int_size]=ws_mcand.flatten()
                vec_ix+=self.int_size
            #now apply the matrix/vector product
            ary_xhat=self.A*ary_xhat)
            #return a list of ws objects (size of x_hat)
            ls_result=[WS(np.zeros(ws_mcand.ary_lowpass.shape),
                             (ws_mcand.one_subband(0)).tup_coeffs)
                             for int_dup in xrange(self.duplicates)]
            vec_ix=0
            for int_dup in xrange(self.duplicates):
                ls_result[int_dup].ws_vector=ary_xhat[vec_ix:vec_ix+self.int_size]
                ls_result[int_dup].unflatten()
                vec_ix+=self.int_size
            ary_mcand=ls_result
        return super(GroupAverage,self).__mul__(ary_mcand)

    class Factory:
        def create(self,ps_parameters,str_section):
            return GroupAverage(ps_parameters,str_section)
