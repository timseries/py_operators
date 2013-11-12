#!/usr/bin/python -tt
import dtcwt
import py_utils.signal_utilities
class dtcwt(Operator):
    """
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for DTCWT
        """
        super(Operator,self).__init__(ps_parameters,str_section)
        
    def __mul__(self,multiplicand):
        """
        Overloading the * operator. multiplicand is:
        forward: a numpy array
        inverse: a wavelet transform object (WS).
        """
        if not self.lgc_adjoint:
            ary_size = size(mutiplicand)
            int_dimension = multiplicand.shape[0]
            if int_dimension==1:
                Yl,Yh = dtwavexfm(multiplicand,self.dict_section['nlevels'], \
                                self.dict_section['biort'],self.dict_section['qshift'])
            elif int_dimension==2:
                Yl,Yh = dtwavexfm2(multiplicand,self.dict_section['nlevels'], \
                                self.dict_section['biort'],self.dict_section['qshift'])
            else:
                Yl,Yh = dtwavexfm3(multiplicand,self.dict_section['nlevels'], \
                                self.dict_section['biort'],self.dict_section['qshift'], \
                                discard_level_1=False)
            multiplicand=self.order_subbands(Yl,Yh,ary_size)
        else:
            ary_size = mutiplicand.ary_size
            int_dimension = mutiplicand.int_dimension
            Yl,Yh = self.reorder_subbands(multiplicand.get_subbands(), \
                                        self.dict_section['nlevels'],ary_size)
            if int_dimension==1:
                multiplicand = dtwaveifm(Yl,Yh, \
                                  self.dict_section['biort'],self.dict_section['qshift'])
            elif int_dimension==2:
                multiplicand = dtwaveifm2(Yl,Yh, \
                                          self.dict_section['biort'], \
                                          self.dict_section['qshift'])
            else:
                multiplicand = dtwavexfm3(multiplicand, \
                                          self.dict_section['biort'], \
                                          self.dict_section['qshift'], \
                                          discard_level_1=False)
        return  super(Operator,self).__mul__(multiplicand)

    def order_subbands(self,Yl,Yh,ary_size)
        """
        Reorder the DTCWT coefficient subbands into cell array 
        for efficient array processing and indexing.
        """
        int_levels = len(Yh)
        if len(Yh[1])==7:
            int_dimension = 3
            subbands_per_level = 28
            ary_size = size(Yl)*2^(int_levels-1)
        else:
            if 