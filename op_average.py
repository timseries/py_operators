#!/usr/bin/python -tt
import numpy as np
from numpy import array, zeros, conj
from numpy import arange as ar
from scipy.sparse import csr_matrix
import cPickle
import os

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
        self.group_type=self.get_val('grouptype',False)
        self.average_type=self.get_val('averagetype',False)
        self.duplicates=None
        self.dimension=None
        self.int_size=None
        self.csr_avg=None
        self.csr_avg_save=None
        self.S=None
        self.L=None
        self.theta=None
        self.file_path=None
        self.initialized=False

    def __mul__(self,mcand):
        """
        Check superclass.
        input is a list of ws objects

        __mul__ supports the following for ws_mcand:
        Forward:
        --------
        1) A list of objects that have the flatten() method.
        2) A K=self.duplicates-sized list of arrays: [Mx1 array,Mx1 array,...].
        3) One big KMx1 array.
        
        Note: A object of type 'WS' needs to be passed to __mul__ if this average object
        hasn't been previously initialized.
        
        Adjoint:
        --------
        For avg_type==cluster:
           1) A single object that has the flatten() method.
           2) One big array Mx1
        For avg_type==group:
           1) A list of objects that have the flatten() method.
           2) A K=self.duplicates-sized list of arrays: [Mx1 array,Mx1 array,...]
           3) One big array KMx1

        Returns a list of arrays (cluster adjoint, group) or a single array (cluster forward).
        """
        
        #if pkl-saved version of A doesn't exist, create one and save    
        mcand_is_list=(mcand.__class__.__name__=='list')
        mcand_el_is_ws=False
        mcand_el_is_ary=False
        mcand_list_sz=0
        if mcand_is_list:
            temp_mcand=mcand[0]
            mcand_list_sz=len(mcand)
        else:    
            temp_mcand=mcand
        if not self.initialized:
            self.init_csr_avg(temp_mcand)
        if mcand_is_list:
            if self.duplicates!=mcand_list_sz:
                raise ValueError('not enough elements in mcand exptected '+ str(self.duplicates) + 
                                     ' got ' + str(mcand_list_sz))
            mcand=su.flatten_list(mcand)
        else:
            if mcand.ndim>1:
                mcand=mcand.flatten()
        if self.lgc_adjoint and self.average_type=='cluster': #adjoint
            mcand=self.csr_avg.transpose()*mcand
        else: #forward,no need to take the transpose for group, since this matrix is symmetric
            mcand=self.csr_avg*mcand
        if (self.average_type=='group' or 
            (self.lgc_adjoint and self.average_type=='cluster')):
            mcand=su.unflatten_list(mcand,self.duplicates)
        return super(Average,self).__mul__(mcand)

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
        if self.group_type=='parentchildren':
            self.duplicates=2
            self.groupsize=2**self.dimension+1
        elif self.group_type=='parentchild':    
            self.duplicates=2**self.dimension+1
            self.groupsize=2
        else:
            raise ValueError('unsupported group_type: ' + self.group_type)
        self.S=ws_mcand.int_subbands
        self.theta=ws_mcand.int_orientations
        self.sparse_matrix_input=self.get_val('sparsematrixinput',False)
        #populate the cluster structure
        self.file_string=(self.average_type+'_'+
                         self.group_type+'_'+
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
        ws_dtype='uint32'
        int_g_count=1 #zero is reserved for absence of group,start at 1
        grp_el_sz=ws_mcand.is_wavelet_complex()+1
        #preallocate the row locations
        ls_ws_data=[(ws_mcand*0).cast(ws_dtype) for j in xrange(self.duplicates)]
        #coarse to fine assignment of group size inverses (1/g_i)
        #and assignemnt of group numbers
        if self.group_type=='parentchildren': 
            #interleave the non-overlapping group number
            #assignments in the two replicated copies            
            for int_dup in xrange(self.duplicates):
                ws_csr_avg_data=ls_ws_data[int_dup]
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
                           ws_csr_avg_data.set_subband(s,1)
                       else:    
                           ws_csr_avg_data.set_subband(s,self.duplicates)
                       #child
                       if (s-self.theta<=self.theta):
                           ws_csr_avg_data.set_subband(s-self.theta,1)
                       else:    
                           ws_csr_avg_data.set_subband(s-self.theta,self.duplicates)
            int_g_count-=1 #the max group count (for just the first orientation at this point)
        elif self.group_type=='parentchild':
            #each entry corresponds to a level and is a dict itself
            #store for fast lookup of children indices from the parent index
            dict_level_indices={} 
            #now assign each group elementwise, slower but the code is clearer
            #we're still assigning the parent (s) and child (s-self.theta) on each iteration
            for s in xrange(self.S-1,self.theta,-self.theta):
                ws_csr_avg_data=ls_ws_data[0]
                int_level,int_orientation=ws_csr_avg_data.lev_ori_from_subband(s)
                int_s_size=ws_csr_avg_data.get_subband(s).size
                ary_s_shape=np.array(ws_csr_avg_data.get_subband(s).shape)
                if not dict_level_indices.has_key(int_level):
                    dict_level_indices[int_level]={}
                dict_child_indices=dict_level_indices[int_level]
                #gather the flattened subbands, there are K replicates
                ls_flat_parents=[ls_ws_data[int_dup].get_subband(s).flatten()
                                 for int_dup in xrange(self.duplicates)]
                ls_flat_children=[ls_ws_data[int_dup].get_subband(s-self.theta).flatten()
                                  for int_dup in xrange(self.duplicates)]
                for int_parent_ix in xrange(int_s_size):
                    int_skip=0
                    for int_child_ix in xrange(2**self.dimension):
                        if not dict_child_indices.has_key(int_child_ix):
                            #get the child indices for this level, if not previously computed
                            #this is accomplished by finding the indices corresponding to each of 
                            #2**dimension corners
                            ary_dummy_child=np.zeros(ary_s_shape*2,dtype=bool)
                            ary_dummy_child[ws_csr_avg_data.ds_slices[int_child_ix]]=1
                            dict_child_indices[int_child_ix]=np.nonzero(ary_dummy_child.flatten())[0]
                        #we need to indroduce an optional offset if at any point
                        #when cycling through the children we encounter an occupied group number
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
                    ls_ws_data[int_dup].set_subband(s,ls_flat_parents[int_dup].reshape(ary_s_shape))
                    ls_ws_data[int_dup].set_subband(s-self.theta,ls_flat_children[int_dup].reshape(2*ary_s_shape))
            #mask the group numbers with the appropriate cluster sizes if we're building a cluster averaging matrix
            if self.average_type=='cluster':    
                for s in xrange(self.S-1,self.theta,-self.theta):
                    ary_s_shape=np.array(ls_ws_data[0].get_subband(s).shape)
                    ls_flat_parents=[ls_ws_data[int_dup].get_subband(s).flatten()
                                     for int_dup in xrange(self.duplicates)]
                    ls_flat_children=[ls_ws_data[int_dup].get_subband(s-self.theta).flatten()
                                      for int_dup in xrange(self.duplicates)]
                    for int_dup in xrange(self.duplicates):
                        cluster_mask=ls_flat_parents[int_dup].reshape(ary_s_shape)>0
                        if (s>=self.S-self.theta):
                            #case 1: cluster size is the number of children at the coarsest level
                            ls_ws_data[int_dup].set_subband(s,(self.duplicates-1)*cluster_mask)
                        elif (s>=2*self.theta):
                            #case 3: the parent and child, when child is finest layer. 
                            #(No overlapped variables for finest layer, so cluster size is 1)
                            ls_ws_data[int_dup].set_subband(s,self.duplicates*cluster_mask)
                            child_cluster_mask=ls_flat_children[int_dup].reshape(2*ary_s_shape)>0
                            ls_ws_data[int_dup].set_subband(s-self.theta,1*child_cluster_mask)
                        else:    
                            #case 2: layers in the middle with no children. The cluster size is the number of children + 1
                            ls_ws_data[int_dup].set_subband(s,self.duplicates*cluster_mask)
            #now that we've done just one tree across a given orientation, we can easily copy this structure
            #to the other subbands
            int_g_count-=1 #the max group count (for just the first orientation at this point)
            if self.average_type=='group':
                int_offset=int_g_count
            else:    
                int_offset=0 #direct copy of the subbands in the same level
            for s in xrange(self.S-1,self.theta-1,-self.theta):
                for int_dup in xrange(self.duplicates):
                    ary_subband=ls_ws_data[int_dup].get_subband(s).copy()
                    for int_ori in xrange(1,self.theta):
                        if self.average_type=='group': #else do direct copy for cluster sizes
                            ary_subband[ary_subband>0]+=int_offset
                        ls_ws_data[int_dup].set_subband(s-int_ori,ary_subband.copy())
            int_g_count*=self.theta #max group count for all orientations
            del dict_child_indices
            del ls_flat_parents
            del ls_flat_children
        else:
            raise ValueError('unsupported group type')            
        #now we can build the sparse matrix...
        if self.average_type=='group':
            #iterate through the group numbers, should have an MXM matrix  at the end
            ary_csr_groups=su.flatten_list(ls_ws_data)
            #next build the group columns and rows    
            ary_csr_groups_ix_tmp=sorted(zip(ary_csr_groups,np.arange(ary_csr_groups.size)))
            ary_csr_groups=np.array([pt[0] for pt in ary_csr_groups_ix_tmp])
            ary_csr_groups_ix=np.array([pt[1] for pt in ary_csr_groups_ix_tmp])
            del ary_csr_groups_ix_tmp
            ary_csr_groups_0_offset=np.nonzero(ary_csr_groups)[0][0]
            #get rid of the indicies which don't correspond to groups
            ary_csr_groups=np.array(ary_csr_groups[ary_csr_groups_0_offset::],dtype='uint32')
            ary_csr_groups_ix=ary_csr_groups_ix[ary_csr_groups_0_offset::]
            #group the indices
            ary_csr_groups_ix=np.array([list(ary_csr_groups_ix[ix_:ix_+self.groupsize]) 
                                        for ix_ in xrange(0,ary_csr_groups_ix.size,self.groupsize)])
            csr_cols=np.tile(ary_csr_groups_ix,self.groupsize).flatten()
            csr_rows=np.repeat(ary_csr_groups_ix.flatten(),self.groupsize)
            csr_data=grp_el_sz*self.groupsize*np.ones(csr_rows.size,'uint8')
            self.csr_avg_save=csr_matrix((csr_data,(csr_rows,csr_cols)),
                                         shape=(self.int_size*self.duplicates,self.int_size*self.duplicates))
            self.csr_avg=csr_matrix((1.0/csr_data,(csr_rows,csr_cols)),
                                    shape=(self.int_size*self.duplicates,self.int_size*self.duplicates))
            del ary_csr_groups
            #end old implementation
        else: #cluster average type
            #build the 'A' matrix in eq 13 and 14
            csr_data=np.array(su.flatten_list(ls_ws_data),dtype='uint8')
            csr_cols=np.nonzero(csr_data)[0]
            csr_data=csr_data[csr_cols]
            csr_rows=csr_cols.copy()
            #wrap the row numbers back
            while np.max(csr_rows)>=self.int_size:
                csr_rows[csr_rows>=self.int_size]-=self.int_size
            self.csr_avg_save=csr_matrix((csr_data,(csr_rows,csr_cols)),
                                         shape=(self.int_size,self.int_size*self.duplicates))
            self.csr_avg=csr_matrix((1.0/csr_data,(csr_rows,csr_cols)),
                                    shape=(self.int_size,self.int_size*self.duplicates))
        del csr_data
        del csr_rows
        del csr_cols
        
    def load_csr_avg(self):
        sec_input=sf.create_section(self.get_params(),self.sparse_matrix_input)
        sec_input.filepath+=self.file_string
        self.file_path=sec_input.filepath
        if os.path.isfile(self.file_path):
            self.csr_avg_save=sec_input.read({},True)
            if self.csr_avg_save!=None:
                self.csr_avg=csr_matrix((1.0/self.csr_avg_save.data,
                                         self.csr_avg_save.indices,
                                         self.csr_avg_save.indptr),
                                         shape=self.csr_avg_save.shape)
                return True
        return False

    def save_csr_avg(self):
        if not self.file_path:
            self.file_path=self.ps_parameters.str_file_dir+self.file_string
        filehandler=open(self.file_path, 'wb')
        cPickle.dump(self.csr_avg_save,filehandler)
        filehandler.close()
        self.csr_avg_save=None

    class Factory:
        def create(self,ps_parameters,str_section):
            return Average(ps_parameters,str_section)
