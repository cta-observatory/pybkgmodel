import numpy
import scipy.special
import scipy.optimize

import astropy.units as u
import astropy.io.fits as pyfits
from astropy.wcs import WCS

from astropy.coordinates import SkyCoord, Angle
from matplotlib import pyplot


def solid_angle_lat_lon_rectangle(theta_E, theta_W, phi_N, phi_S):
    """
    Calculate the solid angle of a latitude-longitude rectangle on a globe.
    Source of the formula: https://en.wikipedia.org/wiki/Solid_angle

    Parameters
    ----------
    phi_N: astropy.units.Quantity or astropy.coordinates.Angle
        North line of latitude
    phi_S: astropy.units.Quantity or astropy.coordinates.Angle
        South line of latitude
    theta_E: astropy.units.Quantity or astropy.coordinates.Angle
        East line of longitude
    theta_W: astropy.units.Quantity or astropy.coordinates.Angle
        West line of longitude

    Returns
    -------
    solid_angle: astropy.units.sr
        Calculated solid angle of a latitude-longitude rectangle.
    """

    phi_N = phi_N.to(u.Unit("rad"))
    phi_S = phi_S.to(u.Unit("rad"))
    theta_E = theta_E.to(u.Unit("rad"))
    theta_W = theta_W.to(u.Unit("rad"))
    solid_angle = (numpy.sin(phi_N) - numpy.sin(phi_S)) * (theta_E.to_value() - theta_W.to_value()) * u.sr

    return solid_angle


def cstat(y, model_y):
    """
    Poissonian C-statistics value.

    Parameters
    ----------
    y: array_like
        Measured counts. Must be integer.
    model_y: array_like
        Predicted counts

    Returns
    -------
    val: float
        2 * log-likelihood value.

    """
    val = -2 * numpy.sum(y * numpy.log(model_y) - model_y - scipy.special.gammaln(y+1))

    return val


def pwl2counts(emin, emax, norm, e0, index):
    """
    Integrated power law spectrum.

    Parameters
    ----------
    emin: array_like
        Minimal energy for integration.
    emax: array_like
        Minimal energy for integration.
    norm: array_like
        Spectral normalization.
    e0: array_like
        Spectrum normalization energy.
    index: array_like
        Spectral index

    Returns
    -------
    counts: array_like
        Integrated spectrum value in the range [emin; emax]

    """
    counts = norm * e0 / (index + 1) * ((emax/e0).decompose()**(index + 1) - (emin/e0).decompose()**(index + 1))
    return counts


def nodespec_integral(energy_edges, dnde):
    """
    Differential node spectrum, integrated within the energy_edges.
    Nodes of the spectrum are assumed to be located
    at sqrt(energy_edges[1:] * energy_edges[:-1]).

    Parameters
    ----------
    energy_edges: array_like of astropy.units.Quantity
        Energy edges of the node spectrum bins.
    dnde: array_like of astropy.units.Quantity
        Differential flux values of the spectrum nodes.

    Returns
    -------
    counts: astropy.units.Quantity
        Integrated spectrum in each of the bins,
        defined by energy_edges.

    """

    if isinstance(energy_edges.unit, u.DexUnit):
        energy_edges = energy_edges.physical

    if isinstance(dnde.unit, u.DexUnit):
        dnde = dnde.physical

    energy = numpy.sqrt(energy_edges[1:] * energy_edges[:-1])
    counts = numpy.zeros(len(energy)) * u.one

    xunit = u.DexUnit(energy_edges.unit)
    yunit = u.DexUnit(dnde.unit)

    dx = numpy.diff(energy.to(xunit).value)
    dy = numpy.diff(dnde.to(yunit).value)
    indicies = dy / dx
    indicies = numpy.concatenate(
        (indicies[:1], indicies, indicies[-1:])
    )

    counts += pwl2counts(
        emin=energy_edges[:-1],
        emax=energy,
        norm=dnde,
        e0=energy,
        index=indicies[:-1]
    )
    counts += pwl2counts(
        emin=energy,
        emax=energy_edges[1:],
        norm=dnde,
        e0=energy,
        index=indicies[1:]
    )

    return counts


def node_cnt_diff(dnde, energy_edges, counts, poisson=False):
    """
    Summed squared difference between the node spectrum
    integral flux and the specified value.
    """
    ncounts = nodespec_integral(energy_edges, dnde)

    if not poisson:
        delta = (counts - ncounts)**2
    else:
        delta = cstat(counts, ncounts)

    return delta.sum()


class CameraImage:
    def __init__(self, counts, xedges, yedges, energy_edges, center=None, mask=None, exposure=None):
        nx = xedges.size - 1
        ny = yedges.size - 1

        if mask is None:
            mask = numpy.ones((nx, ny), dtype=bool)

        if exposure is None:
            exposure = numpy.ones((nx, ny), dtype=numpy.float) * u.s
        elif exposure.shape == ():
            exposure = numpy.repeat(exposure, nx * ny).reshape((nx, ny))

        self.raw_counts = counts
        self.xedges = xedges
        self.yedges = yedges
        self.energy_edges = energy_edges
        self.center = center
        self.mask = mask
        self.raw_exposure = exposure

        self.pixel_coords = self.get_pixel_coords()
        self.pixel_area = self.get_pixel_areas()

    @classmethod
    def from_events(cls, event_file, xedges, yedges, energy_edges):
        center = cls.get_poiting(event_file)
        image = cls.bin_events(event_file, xedges, yedges, energy_edges)

        return cls(image, xedges, yedges, energy_edges, center=center, exposure=event_file.events.eff_obs_time)

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

    def get_pixel_coords(self):
        pass

    def get_pixel_areas(self):
        pass

    @classmethod
    def get_poiting(cls, event_file):
        return SkyCoord(ra=event_file.pointing_ra.mean(), dec=event_file.pointing_dec.mean())

    @property
    def counts(self):
        return self.raw_counts * self.mask

    @property
    def exposure(self):
        return self.raw_exposure * self.mask

    @property
    def rate(self):
        return self.counts / self.raw_exposure / self.pixel_area
    

    def _spectrum(self, index):
        """
        Calclate the energy spectrum assuming the power law spectral shape 
        dN/dE = A*(E/E0)**index with the specified
        spectral index. Rate is calculated in at e0 = (emin * emax)**0.5
        following the existing energy binning.
        """
        emin = self.energy_edges[:-1]
        emax = self.energy_edges[1:]
        e0 = (emin * emax)**0.5
        int2diff = (index + 1) / e0 / ((emax/e0).decompose()**(index + 1) - (emin/e0).decompose()**(index + 1))
        return int2diff

    def differential_rate(self, index=None):
        """
        Differential count rate assuming the power law spectral shape

        Parameters
        ----------
        index: float
            Power law spectral index to assume.

        Returns
        -------
        differential_rate: array_like astropy.unit.Quantity
            Computed rate of the same shape as the camera image.
        """

        int2diff = self._spectrum(index)
        dnde = self.rate * int2diff[:, None, None]

        return dnde

    def mask_half(self, pointer):
        """Excludes the half of the camera containing the sources.

        Parameters
        ----------
        pointer : SkyCoord
            Source position in the camera coordinate system.
        """
        offset_delta = Angle('90d')

        pixel_position_angles = self.center.position_angle(self.pixel_coords)
        pointer_position_angle = self.center.position_angle(pointer)
        position_angle_offest = (pixel_position_angles - pointer_position_angle).wrap_at('180d')

        to_mask = (position_angle_offest >= -offset_delta) & (position_angle_offest < offset_delta)

        self.mask[to_mask] = False

    def mask_region(self, region):
        """Masks (empties) all camera pixel contained in the provided sky region.

        Parameters
        ----------
        region : region object
            See https://astropy-regions.readthedocs.io/en/stable/contains.html
        """
        dummy_wcs = WCS(naxis=2)
        # Taken from https://docs.astropy.org/en/stable/wcs/example_create_imaging.html
        # Set up an "Airy's zenithal" projection
        # Vector properties may be set with Python lists, or np arrays
        dummy_wcs.wcs.crpix = [-234.75, 8.3393]
        dummy_wcs.wcs.cdelt = numpy.array([-0.066667, 0.066667])
        dummy_wcs.wcs.crval = [0, -90]
        dummy_wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]
        dummy_wcs.wcs.set_pv([(2, 1, 45.0)])

        in_region = region.contains(self.pixel_coords, dummy_wcs)
        self.mask[in_region] = False

    def mask_reset(self):
        """_summary_
        """
        self.mask = numpy.ones((self.xedges.size - 1, self.yedges.size - 1), dtype=bool)


    def plot(self, energy_bin_id=0, ax_unit='deg', val_unit='1/s', **kwargs):
        pyplot.xlabel(f'X [{ax_unit}]')
        pyplot.ylabel(f'Y [{ax_unit}]')
        pyplot.pcolormesh(
            self.xedges.to(ax_unit).value,
            self.yedges.to(ax_unit).value,
            (self.counts[energy_bin_id] / self.raw_exposure).to(val_unit).transpose(),
            **kwargs
        )
        pyplot.colorbar(label=f'rate [{val_unit}]')


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

    def get_pixel_coords(self):
        if self.center is None:
            frame=None
        else:
            frame=self.center.skyoffset_frame()

        x = (self.xedges[1:] + self.xedges[:-1]) / 2
        y = (self.yedges[1:] + self.yedges[:-1]) / 2
        xx, yy = numpy.meshgrid(x, y, indexing='ij')

        pixel_coords = SkyCoord(
            xx,
            yy,
            frame=frame
        )

        return pixel_coords

    def get_pixel_areas(self):
        nx = self.xedges.size - 1
        ny = self.yedges.size - 1

        area = numpy.zeros((nx, ny)) * u.sr

        for i in range(nx):
            for j in range(ny):
                area[i, j] = solid_angle_lat_lon_rectangle(self.xedges[i], self.xedges[i+1], self.yedges[j], self.yedges[j+1])

        return area

    def to_hdu(self, name='BACKGROUND'):
        energ_lo = self.energy_edges[:-1]
        energ_hi = self.energy_edges[1:]

        detx_lo = self.xedges[:-1]
        detx_hi = self.xedges[1:]

        dety_lo = self.yedges[:-1]
        dety_hi = self.yedges[1:]

        bkg_rate = self.differential_rate(index=-2)
        
        counts = self.counts
        raw_exposure = self.raw_exposure
        pixel_area = self.pixel_area
        int2diff = self._spectrum(index=-2)

        col_energ_lo = pyfits.Column(name='ENERG_LO', unit='TeV', format=f'{energ_lo.size}E', array=[energ_lo])
        col_energ_hi = pyfits.Column(name='ENERG_HI', unit='TeV', format=f'{energ_hi.size}E', array=[energ_hi])
        col_detx_lo = pyfits.Column(name='DETX_LO', unit='deg', format=f'{detx_lo.size}E', array=[detx_lo])
        col_detx_hi = pyfits.Column(name='DETX_HI', unit='deg', format=f'{detx_hi.size}E', array=[detx_hi])
        col_dety_lo = pyfits.Column(name='DETY_LO', unit='deg', format=f'{dety_lo.size}E', array=[dety_lo])
        col_dety_hi = pyfits.Column(name='DETY_HI', unit='deg', format=f'{dety_hi.size}E', array=[dety_hi])

        col_bkg_rate = pyfits.Column(
            name='BKG',
            unit='s^-1 MeV^-1 sr^-1',
            format=f"{bkg_rate.size}E",
            array=[
                bkg_rate.to('1 / (s * MeV * sr)').value.transpose()
            ],
            dim=str(bkg_rate.shape))

        cols = []
        for key, arr, unit in zip(
            ['COUNTS', 'EXPOSURE', 'PIXAREA', 'SPECTRUM'],
            [counts, raw_exposure, pixel_area, int2diff],
            ['', 's', 'sr', 'MeV-1'],
        ):
            if isinstance(arr, u.Quantity):
                arr = arr.to_value(unit)
            col = pyfits.Column(
                name=key,
                unit=unit,
                format=f"{arr.size}E",
                array=[arr.transpose()],
                dim=str(arr.shape)
            )
            cols.append(col)

        columns = [
            col_energ_lo,
            col_energ_hi,
            col_detx_lo,
            col_detx_hi,
            col_dety_lo,
            col_dety_hi,
            col_bkg_rate,
            *cols,
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
