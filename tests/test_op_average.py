import numpy as np
from py_utils.parameter_struct import ParameterStruct
from py_utils.section_factory import SectionFactory as sf
from py_operators.operator_comp import OperatorComp
ps_path='/home/tim/repos/py_solvers/application/deconvolution/config/uniform_40db_bsnr_cameraman_msistg.ini'
ps_params = ParameterStruct(ps_path)
G=sf.create_section(ps_params,'Transform3')
W = OperatorComp(ps_params,'TransformArray1').ls_ops[0]
wcand=W*np.ones((32,32))
G.init_csr_avg(wcand)