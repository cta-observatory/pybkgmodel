import numpy
import numpy.ma

import astropy.units as u
import astropy.io.fits as pyfits

from astropy.coordinates import SkyCoord, Angle
from matplotlib import pyplot


def rectangle_area(l, w):
    """
    Area of the rectangle on a sphere of the unit radius.

    Parameters
    ----------
    l: astropy.units.rad or convertable to it
        Rectangle extension in longitude.
    w: astropy.units.rad or convertable to it
        Rectangle extension in latitude.

    Returns
    -------
    area: astropy.units.sr
        Calcuated area

    References
    ----------
    [1] https://math.stackexchange.com/questions/1205927/how-to-calculate-the-area-covered-by-any-spherical-rectangle
    [2] http://en.wikipedia.org/wiki/Spherical_trigonometry#Area_and_spherical_excess
    """

    t1 = numpy.tan(l.to('rad').value / 2)
    t2 = numpy.tan(w.to('rad').value / 2)

    return 4 * numpy.arcsin(t1 * t2) * u.sr


def pixel_area(xedges, yedges):
    """
    Area of a rectangular pixel on a shere. Pixel is defined by its edges.

    Parameters
    ----------
    xedges: array_like of astropy.units.rad or convertable to it
        Longitude of the pixel edges. Must have the shape of (2,).
    yedges: array_like of astropy.units.rad or convertable to it
        latitude of the pixel edges. Must have the shape of (2,).

    Returns
    -------
    area: astropy.units.sr
        Calcuated area
    """

    l = abs(xedges[1] - xedges[0])
    w_outer = 2 * max(numpy.abs(yedges))
    w_inner = 2 * min(abs(yedges))

    w_sign = numpy.sign(yedges)
    signes_match = numpy.equal(*w_sign)

    if signes_match:
        area = 0.5 * (rectangle_area(l, w_outer) - rectangle_area(l, w_inner))
    else:
        area = 0.5 * (rectangle_area(l, w_outer) + rectangle_area(l, w_inner))

    return area


class CameraImage:
    def __init__(self, image, xedges, yedges, energy_edges, center=None, mask=None, exposure=None):
        nx = xedges.size - 1
        ny = yedges.size - 1

        if mask is None:
            mask = numpy.ones((nx, ny), dtype=numpy.bool)

        if exposure is None:
            exposure = numpy.ones((nx, ny), dtype=numpy.float) * u.s
        elif isinstance(exposure, float):
            exposure = numpy.repeat(exposure, nx * ny).reshape((nx, ny))

        self.raw_image = image
        self.xedges = xedges
        self.yedges = yedges
        self.energy_edges = energy_edges
        self.center = center
        self.mask = mask
        self.raw_exposure = exposure

        if self.center is None:
            frame=None
        else:
            frame=self.center.skyoffset_frame()

        x = (xedges[1:] + xedges[:-1]) / 2
        y = (yedges[1:] + yedges[:-1]) / 2
        xx, yy = numpy.meshgrid(x, y, indexing='ij')

        self.pixel_coords = SkyCoord(
            xx,
            yy,
            frame=frame
        )

        self.pixel_area = numpy.zeros((nx, ny)) * u.sr
        for i in range(nx):
            for j in range(ny):
                self.pixel_area[i, j] = pixel_area(xedges[i:i+2], yedges[j:j+2])

    @classmethod
    def from_events(cls, event_file, xedges, yedges, energy_edges):
        center = cls.get_poiting(event_file)
        image = cls.bin_events(event_file, xedges, yedges, energy_edges)

        return CameraImage(image, xedges, yedges, energy_edges, center=center)

    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'Center':.<20s}: {self.center}
    {'X range':.<20s}: [{self.xedges.min():.1f}, {self.xedges.max():.1f}]
    {'Y range':.<20s}: [{self.yedges.min():.1f}, {self.yedges.max():.1f}]
    {'X bins':.<20s}: {len(self.xedges) - 1}
    {'X bins':.<20s}: {len(self.yedges) - 1}
    {'Exposure (mean)':.<20s}: {self.raw_exposure[self.mask].mean()}
"""
        )

        return super().__repr__()

    @classmethod
    def bin_events(cls, event_file, xedges, yedges, energy_edges):
        pass

    @classmethod
    def get_poiting(cls, event_file):
        return SkyCoord(ra=event_file.pointing_ra.mean(), dec=event_file.pointing_dec.mean())

    @property
    def image(self):
        return self.raw_image * self.mask

    @property
    def exposure(self):
        return self.raw_exposure * self.mask

    @property
    def rate(self):
        return self.image / self.raw_exposure / self.pixel_area

    def mask_reset(self):
        self.mask = numpy.ones((self.xedges.size - 1, self.yedges.size - 1), dtype=numpy.bool)

    def mask_half(self, pointer):
        offset_delta = Angle('90d')

        pixel_position_angles = self.center.position_angle(self.pixel_coords)
        pointer_position_angle = self.center.position_angle(pointer)
        position_angle_offest = (pixel_position_angles - pointer_position_angle).wrap_at('180d')

        to_mask = (position_angle_offest >= -offset_delta) & (position_angle_offest < offset_delta)

        self.mask[to_mask] = False

    def mask_circle(self, center, rad):
        to_mask = center.separation(self.pixel_coords) < rad
        self.mask[to_mask] = False

    def plot(self, energy_bin_id=0, ax_unit='deg', val_unit='1/s', **kwargs):
        pyplot.xlabel(f'X [{ax_unit}]')
        pyplot.ylabel(f'Y [{ax_unit}]')
        pyplot.pcolormesh(
            self.xedges.to(ax_unit).value,
            self.yedges.to(ax_unit).value,
            (self.image[energy_bin_id] / self.raw_exposure).to(val_unit).transpose()
        )
        pyplot.colorbar(label=f'rate [{val_unit}]')

    def to_hdu(self, name='BACKGROUND'):
        energ_lo = self.energy_edges[:-1]
        energ_hi = self.energy_edges[1:]

        detx_lo = self.xedges[:-1]
        detx_hi = self.xedges[1:]

        dety_lo = self.xedges[:-1]
        dety_hi = self.xedges[1:]

        col_energ_lo = pyfits.Column(name='ENERG_LO', unit='TeV', format=f'{energ_lo.size}E', array=[energ_lo])
        col_energ_hi = pyfits.Column(name='ENERG_HI', unit='TeV', format=f'{energ_hi.size}E', array=[energ_hi])
        col_detx_lo = pyfits.Column(name='DETX_LO', unit='deg', format=f'{detx_lo.size}E', array=[detx_lo])
        col_detx_hi = pyfits.Column(name='DETX_HI', unit='deg', format=f'{detx_hi.size}E', array=[detx_hi])
        col_dety_lo = pyfits.Column(name='DETY_LO', unit='deg', format=f'{dety_lo.size}E', array=[dety_lo])
        col_dety_hi = pyfits.Column(name='DETY_HI', unit='deg', format=f'{dety_hi.size}E', array=[dety_hi])

        col_bkg_rate = pyfits.Column(
            name='BKG',
            unit='s^-1 MeV^-1 sr^-1',
            format=f"{self.rate.size}E",
            # TODO: add proper unit convertion here
            array=[
                self.rate.to('1 / (s * sr)').value.transpose()
            ],
            dim=str(self.image.shape))

        columns = [
            col_energ_lo,
            col_energ_hi,
            col_detx_lo,
            col_detx_hi,
            col_dety_lo,
            col_dety_hi,
            col_bkg_rate
        ]

        col_defs = pyfits.ColDefs(columns)
        hdu = pyfits.BinTableHDU.from_columns(col_defs)
        hdu.name = name

        hdu.header['HDUDOC'] = 'https://github.com/open-gamma-ray-astro/gamma-astro-data-formats'
        hdu.header['HDUVERS'] = '0.2'
        hdu.header['HDUCLASS'] = 'GADF'
        hdu.header['HDUCLAS1'] = 'RESPONSE'
        hdu.header['HDUCLAS2'] = 'BKG'
        hdu.header['HDUCLAS3'] = 'FULL-ENCLOSURE'
        hdu.header['HDUCLAS4'] = 'BKG_3D'

        return hdu


class RectangularCameraImage(CameraImage):
    @classmethod
    def bin_events(cls, event_file, xedges, yedges, energy_edges):
        center = cls.get_poiting(event_file)
        events = SkyCoord(ra=event_file.event_ra, dec=event_file.event_dec)

        cam = events.transform_to(center.skyoffset_frame())

        hist, _ = numpy.histogramdd(
            sample=(
                event_file.event_energy,
                cam.lon,
                cam.lat
            ),
            bins=(
                energy_edges,
                xedges,
                yedges
            )
        )

        return hist
