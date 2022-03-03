import numpy as np
from numpy.testing import assert_almost_equal
from astropy import units as u

def test_cone_solid_angle():
    from pybkgmodel.utils import cone_solid_angle
    c = cone_solid_angle(180*u.deg)
    assert_almost_equal(c.value, 4*np.pi)
