import os
from os.path import join as pjoin
from nose.tools import raises
from nose.plugins.attrib import attr

import numpy as np

from py_operators.op_blur import Blur
from py_utils.parameter_struct import ParameterStruct
from py_utils.section_factory import SectionFactory as sf

from util import assert_almost_equal, summarise_mat, summarise_cube
from util import assert_percentile_almost_equal, assert_almost_equal
import dtcwt.tests.datasets as datasets

TOLERANCE = 1e-5

def assert_almost_equal_to_summary(a, summary, *args, **kwargs):
    assert_almost_equal(summarise_mat(a), summary, *args, **kwargs)

def assert_percentile_almost_equal_to_summary(a, summary, *args, **kwargs):
    assert_percentile_almost_equal(summarise_mat(a), summary, *args, **kwargs)

def assert_almost_equal_to_summary_cube(a, summary, *args, **kwargs):
    assert_almost_equal(summarise_cube(a), summary, *args, **kwargs)

def assert_percentile_almost_equal_to_summary_cube(a, summary, *args, **kwargs):
    assert_percentile_almost_equal(summarise_cube(a), summary, *args, **kwargs)

def setup():

    this_dir = os.path.dirname(__file__)

    global qbgn
    qbgn = np.load(pjoin(this_dir, 'qbgn.npz'))['qbgn']

    global verif
    verif = np.load(pjoin(this_dir, 'verification.npz'))

    global ps_params
    ps_params = ParameterStruct(pjoin(this_dir, 'testparameters.ini'))

    global cell_image
    cell_image = sf.create_section(ps_params,'Input1').read({},True)

def test_lena_loaded():
    assert lena.shape == (512, 512)
    assert lena.min() >= 0
    assert lena.max() <= 1
    assert lena.dtype == np.float32

def test_lena_loaded():
    assert verif is not None

def test_blur_2D():
    ls_blurs=['Modality1_1','Modality1_2']
    for idx, str_blur in enumerate(ls_blurs):
        H = Blur(ps_params,str_blur)
        assert_almost_equal(H * lena, verif['lena_blur_2D_{0}'.format(idx)], tolerance=TOLERANCE)

def test_implicit_blur_3D():
    #test the blurring of the implicit convolution (3D) psf
    ls_blurs=['Modality1_3']

    for idx, str_blur in enumerate(ls_blurs):
        H = Blur(ps_params,str_blur)
        assert_almost_equal(H * cell_image, verif['cell_blur_3D_{0}'.format(idx)], tolerance=TOLERANCE)
