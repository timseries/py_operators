#!/usr/bin/python -tt
import numpy as np

from dtcwt.numpy.common import Pyramid

from py_utils.signal_utilities import ws as ws, Scat
from py_utils.section_factory import SectionFactory as sf
from py_operators.operator import Operator

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
        self.depth = self.get_val('depth',False)
        self.transform_sec = self.get_val('transform',False)
        self.W = sf.create_section(ps_params,self.transform_sec)
        #create the versions of W we'll need in a list
        self.max_transform_levels = self.W.get_val('nlevels')
        self.W = []
        for j in xrange(self.max_transform_levels+1,0,-1):
            ps_params.set_key_val_pairs(self.transform_sec,['nlevels'],[j])
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
            for level in xrange(self.depth):
                if level==0:
                    parent_nodes = [Node(W[0]*multiplicand)]
                    S = Scat(parent_nodes[0],parent_nodes[0].int_level,self.depth)
                    int_orientations = parent_nodes[0].int_orientations
                else:
                    parent_nodes_next = []
                    for parent_node in parent_nodes:
                        #generate nodes at this level for each parent
                        parent_mod = parent_node.modulus()
                        child_nodes = []
                        for subband_index in xrange(1,parent_mod.int_subbands):
                            # subband_label = np.mod(child-1,int_orientations)+1
                            num_levels = parent_mod.int_levels
                            if subband_index <= wstemp_mod.int_subbands-wstemp_mod.int_orientations:
                                w_index = num_levels-subband_index/int_orientations
                                parent_nodes_next.append(Node(W[w_index]*parent_mod.get_subband(subband_index)))
                                child_nodes.append(parent_nodes_next[-1])
                            else:
                                child_nodes.append(Node(upsample(wstemp_mod.get_subband(child))))
                        #we've generated all of the nodes for parent_node_mod now, we can delete
                        #parent_node_mod's ws object and move on to the next one
                        parent_node.data = deepcopy(parent_mod.ary_scaling)
                        del parent_mod
                        parent_node.children = child_nodes
                        parent_node.delete_wrapped_instance()
                    parent_nodes = parent_nodes_next
                    
        else:#adjoint, multiplicand should be a WS object
            print 'inverse scattering not implemented yet'
                
        return super(Scattering,self).__mul__(multiplicand)

    class Factory:
        def create(self,ps_parameters,str_section):
            return Scattering(ps_parameters,str_section)
