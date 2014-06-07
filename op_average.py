#!/usr/bin/python -tt
import numpy as np
from numpy import array, zeros, conj
from numpy import arange as ar
from scipy.sparse import csr_matrix
import cPickle

from py_utils.signal_utilities.ws import WS
import py_utils.signal_utilities.sig_utils as su
from py_operators.operator import Operator
from py_utils.section_factory import SectionFactory as sf

import pdb

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

    def create_csr_avg(self,ws_mcand):
        ws_mcand.flatten()
        #for group averaging 
        #determine the size (grp_el_sz) of each element, used in average energy computation
        if self.average_type=='group':
            grp_el_sz=ws_mcand.is_wavelet_complex()+1
            int_g_count=1 #zero is reserved for absence of group,start at 1
        #preallocate the row locations
        ls_ws_csr_avg_data=[WS(np.zeros(ws_mcand.ary_lowpass.shape),
                           (ws_mcand.one_subband(0)).tup_coeffs).cast('float32')
                           for j in xrange(self.duplicates)]
        #coarse to fine assignment of group size inverses (1/g_i)
        #and assignemnt of group numbers
        if self.grouptype=='parentchildren': 
            #interleave the non-overlapping group number
            #assignments in the two replicated copies            
            for int_dup in xrange(self.duplicates):
                ws_csr_avg_data=ls_ws_csr_avg_data[int_dup]
                int_s_start=self.S-1-self.theta*int_dup
                parent_indices=[ar(int_s_start-2*j*self.theta,
                                   int_s_start-2*j*self.theta-self.theta,-1) 
                                   for j in xrange(int_s_start/self.theta/2)]
                parent_indices=np.hstack(parent_indices)
                for s in parent_indices:
                    if self.average_type=='group':
                        int_s_size=ws_csr_avg_data.get_subband(s).size
                        tup_s_shape=ws_csr_avg_data.get_subband(s).shape
                        group_indices=ar(int_g_count,int_g_count+int_s_size).reshape(tup_s_shape)
                        ws_csr_avg_data.set_subband(s,group_indices)#parent
                        ws_csr_avg_data.set_subband(s-self.theta,ws_csr_avg_data.get_upsampled_parent(s-self.theta))#child
                        int_g_count+=int_s_size
                    else: #we doing a cluster averaging matrix
                       #assign the parent and children on each iteration, as with the group variant 
                       #parent
                       if (s>=self.S-self.theta):
                           ws_csr_avg_data.set_subband(s,1.0)
                       else:    
                           ws_csr_avg_data.set_subband(s,1.0/self.duplicates)
                       #child
                       if (s-self.theta<=self.theta):
                           ws_csr_avg_data.set_subband(s-self.theta,1.0)
                       else:    
                           ws_csr_avg_data.set_subband(s-self.theta,1.0/self.duplicates)
                           
        elif self.grouptype=='parentchild':
            #each entry corresponds to a level and is a dict itself
            #store for fast lookup of children indices from the parent index
            dict_level_indices={} 
            #now assign each group elementwise, slower but the code is clearer
            int_g_count=1
            #we're still assigning the parent (s) and child (s-self.theta) on each iteration
            for s in xrange(self.S-1,self.theta-1,-1):
                ws_csr_avg_data=ls_ws_csr_avg_data[0]
                int_level,int_orientation=ws_csr_avg_data.lev_ori_from_subband(s)
                int_s_size=ws_csr_avg_data.get_subband(s).size
                tup_s_shape=ws_csr_avg_data.get_subband(s).shape
                if not dict_child_indices.has_key(int_level):
                    dict_level_indices[int_level]={}
                dict_child_indices=dict_level_indices[int_level]
                #gather the flattened subbands, there are K replicates
                ls_flat_parents=[ls_ws_csr_avg_data[int_dup].get_subband(s).flatten()
                                 for int_dup in xrange(self.duplicates)]
                ls_flat_children=[ls_ws_csr_avg_data[int_dup].get_subband(s-self.theta).flatten()
                                  for int_dup in xrange(self.duplicates)]
                for int_parent_ix in xrange(int_s_size):
                    int_skip=0
                    for int_child_ix in xrange(2**self.dimension):
                        if not dict_child_indices.has_key(int_child_ix):
                            #get the child indices 
                            ary_dummy_child=np.zeros(tup_s_shape*2,dtype=bool)
                            ary_dummy_child[ws_csr_avg_data.ds_slices[int_child_ix]]=1
                            dict_child_indices[int_child_ix]=np.nonzero(ary_dummy_child.flatten())
                        if ls_flat_parents[int_child_ix][int_parent_ix]>0:
                            int_skip=1
                        ls_flat_parents[int_child_ix+int_skip][int_parent_ix]=int_g_count
                        int_ls_ws_ix=int_child_ix+1+int_skip
                        if int_ls_ws_ix>=self.duplicates:
                            int_ls_ws_ix=0 #wrap
                        ls_flat_children[int_ls_ws_ix][dict_child_indices[int_child_ix][int_parent_ix]]=int_g_count
                        int_g_count+=1
                #store group numbers or cluster inverse sises back in the ws structures, for this s
                for int_dup in xrange(self.duplicates):
                    if self.average_type=='group':
                        ls_ws_csr_avg_data[int_dup].set_subband(s,ls_flat_parents[int_dup].reshape(tup_s_shape))
                        ls_ws_csr_avg_data[int_dup].set_subband(s-self.theta,ls_flat_children[int_dup].reshape(2*tup_s_shape))
                    else:
                        if (s>=self.S-self.theta):
                            ls_ws_csr_avg_data[int_dup].set_subband(s,1.0/(self.duplicates-1))
                        else:    
                            ls_ws_csr_avg_data[int_dup].set_subband(s,1.0/(self.duplicates))
                        if (s-self.theta<=self.theta):
                            ls_ws_csr_avg_data[int_dup].set_subband(s-self.theta,1.0)
                        else:
                            ls_ws_csr_avg_data[int_dup].set_subband(s-self.theta,1.0/(self.duplicates))    
            del dict_child_indices
            del ls_flat_parents
            del ls_flat_children
        else:
            raise ValueError('unsupported group type')            
        #now we can build the sparse matrix...
        if self.average_type=='group':
            #iterate through the group numbers, should have an MXM matrix  at the end
            ary_csr_groups=su.flatten_list(ls_ws_csr_avg_data)
            del ls_ws_csr_avg_data    
            #next build the group columns and rows    
            csr_rows=np.zeros(0,)
            csr_cols=np.zeros(0,)
            for int_group_index in xrange(1,int_g_count):
                grp_el_locs=np.nonzero(ary_csr_groups==int_group_index)[0]
                if grp_el_locs.size>0:
                    csr_cols=np.hstack((csr_cols,np.tile(grp_el_locs,len(grp_el_locs))))
                    csr_rows=np.hstack((csr_rows,np.repeat(grp_el_locs,len(grp_el_locs))))
            csr_data=1.0/(grp_el_sz*self.groupsize*np.ones(csr_rows.size,))
            self.csr_avg=csr_matrix((csr_data,(csr_rows,csr_cols)),
                                    shape=(self.int_size*self.duplicates,self.int_size*self.duplicates))
            del ary_csr_groups
        else: #cluster average type
            #build the 'A' matrix in eq 13 and 14
            # csr_cols=np.arange(0,self.int_size*self.duplicates)
            # csr_rows=np.tile(np.arange(0,self.int_size),self.duplicates)
            csr_data=su.flatten_list(ls_ws_csr_avg_data)
            csr_cols=np.nonzero(csr_data)[0]
            csr_data=csr_data[csr_cols]
            csr_rows=csr_cols.copy()
            #wrap the row numbers back
            while np.max(csr_rows)>=self.int_size:
                csr_rows[csr_rows>=self.int_size]-=self.int_size
            self.csr_avg=csr_matrix((csr_data,(csr_rows,csr_cols)),
                                    shape=(self.int_size,self.int_size*self.duplicates))
        del csr_data
        del csr_rows
        del csr_cols

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
