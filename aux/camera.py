import numpy
import numpy.ma

import astropy.units as u
import astropy.io.fits as pyfits

from astropy.coordinates import SkyCoord, Angle
from matplotlib import pyplot


class CameraImage:
    def __init__(self, image, xedges, yedges, energy_edges, center=None, mask=None, exposure=None):
        if mask is None:
            mask = numpy.ones((xedges.size - 1, yedges.size - 1), dtype=numpy.bool)

        if exposure is None:
            exposure = numpy.ones((xedges.size - 1, yedges.size - 1), dtype=numpy.float) * u.s
        elif isinstance(exposure, float):
            nx = xedges.size - 1
            ny = yedges.size - 1
            exposure = numpy.repeat(exposure, nx * ny).reshpae((nx, ny))

        self.raw_image = image
        self.xedges = xedges
        self.yedges = yedges
        self.energy_edges = energy_edges
        self.center = center
        self.mask = mask
        self.raw_exposure = exposure

        x = (xedges[1:] + xedges[:-1]) / 2
        y = (yedges[1:] + yedges[:-1]) / 2

        xx, yy = numpy.meshgrid(x, y, indexing='ij')

        if self.center is None:
            frame=None
        else:
            frame=self.center.skyoffset_frame()

        self.pixel_coords = SkyCoord(
            xx,
            yy,
            frame=frame
        )

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

        bkg_rate = self.image / self.raw_exposure

        col_bkg_rate = pyfits.Column(
            name='BKG',
            unit='s^-1 MeV^-1 sr^-1',
            format=f"{self.image.size}E",
            # TODO: add proper unit convertion here
            array=[
                bkg_rate.value.transpose()
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
