#!/usr/bin/python -tt
import numpy as np
from copy import deepcopy

from dtcwt.numpy.common import Pyramid

from py_utils.signal_utilities import ws as ws
from py_utils.signal_utilities.sig_utils import upsample
from py_utils.section_factory import SectionFactory as sf
from py_utils.node import Node
from py_operators.operator import Operator

import pdb

class Scattering(Operator):
    """
    Operator which performs the forward/inverse(~) Scattering transform attributed to Mallat.
    Returns a WS object (forward), or a numpy array (inverse)
    """
    
    def __init__(self,ps_params,str_section):
        """
        Class constructor for Scattering
        """
        super(Scattering,self).__init__(ps_params,str_section)
        self.depth = self.get_val('depth',True)
        self.transform_sec = self.get_val('transform',False)
        self.W = sf.create_section(ps_params,self.transform_sec)
        self.max_transform_levels = self.W.nlevels
        if self.depth > self.max_transform_levels:
            ValueError('cannot have more scattering transform laters than wavelet transform scales')
        #create the versions of W we'll need in a list
        self.W = []
        for J in xrange(1,self.max_transform_levels+1):
            ps_params.set_key_val_pairs(self.transform_sec,['nlevels'],[J])
            self.W.append(sf.create_section(ps_params,self.transform_sec))
        
    def __mul__(self,multiplicand):
        """
        Overloading the * operator. multiplicand is{
        forward: a numpy array
        adjoint: a scattering object.}
        """        
        W = self.W
        if not self.lgc_adjoint:
            #inital wavelet transform
            for level in xrange(self.depth+1):
                if level==0:
                    parent_nodes = [Node(W[-1]*multiplicand)]
                    root_node = parent_nodes[-1]
                    int_orientations = parent_nodes[0].int_orientations
                else:
                    parent_nodes_next = []
                    for parent_node in parent_nodes:
                        #generate the modulus for the previous level
                        parent_mod = parent_node.modulus()
                        #only propagate if we're before the scattering transform depth
                        if level<self.depth:
                            child_nodes = []
                            for subband_index in xrange(1,parent_mod.int_subbands):
                                num_levels = parent_mod.int_levels
                                if subband_index < parent_mod.int_subbands-parent_mod.int_orientations:
                                    w_index = num_levels-(subband_index-1)/int_orientations-2
                                    parent_nodes_next.append(Node(W[w_index]*parent_mod.get_subband(subband_index)))
                                    child_nodes.append(parent_nodes_next[-1])
                                else:
                                    child_nodes.append(Node(object))
                                    child_nodes[-1].set_data(upsample(parent_mod.get_subband(subband_index)))
                            parent_node.set_children(child_nodes) #end propagation block
                        #we've generated all of the nodes for parent_node_mod now, we can delete
                        #parent_node_mod's ws object and move on to the next one
                        parent_node.set_data(deepcopy(parent_mod.ary_lowpass))
                        del parent_mod
                        parent_node.delete_wrapped_instance()
                    parent_nodes = parent_nodes_next
            multiplicand = root_node        
        else:#adjoint, multiplicand should be a WS object
            print 'inverse scattering not implemented yet'
        return super(Scattering,self).__mul__(multiplicand)

    class Factory:
        def create(self,ps_parameters,str_section):
            return Scattering(ps_parameters,str_section)
