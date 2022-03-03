import numpy as np
from astropy import units as u

def cone_solid_angle(angle):
    """
    Calculate the solid angle of a view cone.
    Parameters
    ----------
    angle: astropy.units.Quantity or astropy.coordinates.Angle
        Opening angle of the view cone.
    Returns
    -------
    solid_angle: astropy.units.Quantity
        Solid angle of a view cone with opening angle ``angle``.
    """
    angle = angle.to(u.Unit("rad"))
    solid_angle = 2 * np.pi * (1 - np.cos(angle)) * u.sr
    return solid_angle


def cone_solid_angle_rectangular_pyramid(a, b):
    """
    Calculate the solid angle of a view cone.
    Parameters
    ----------
    angle: astropy.units.Quantity or astropy.coordinates.Angle
        Opening angle of the view cone.
    Returns
    -------
    solid_angle: astropy.units.Quantity
        Apex angles of a retangluar pyramid.
    """
    a = a.to(u.Unit("rad"))
    b = b.to(u.Unit("rad"))
    solid_angle = 4 * np.arcsin (np.sin(a/2) * np.sin(b/2)).value * u.sr
    return solid_angle
