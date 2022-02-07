import numpy
import numpy.ma

from astropy.coordinates import SkyCoord, Angle
from matplotlib import pyplot


class CameraImage:
    def __init__(self, image, xedges, yedges, energy_edges, center=None):
        self.image = image
        self.xedges = xedges
        self.yedges = yedges
        self.energy_edges = energy_edges
        self.center = center

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
"""
        )

        return super().__repr__()

    @classmethod
    def bin_events(cls, event_file, xedges, yedges, energy_edges):
        pass

    @classmethod
    def get_poiting(cls, event_file):
        return SkyCoord(ra=event_file.pointing_ra.mean(), dec=event_file.pointing_dec.mean())

    def remove_mask(self):
        self.image.mask = False

    def mask_half(self, pointer):
        offset_delta = Angle('90d')

        pixel_position_angles = self.center.position_angle(self.pixel_coords)
        pointer_position_angle = self.center.position_angle(pointer)
        position_angle_offest = (pixel_position_angles - pointer_position_angle).wrap_at('180d')

        to_mask = (position_angle_offest >= -offset_delta) & (position_angle_offest < offset_delta)

        self.image = numpy.ma.masked_where(to_mask[None, ...], self.image)

        return position_angle_offest

    def mask_circle(self, center, rad):
        to_mask = center.separation(self.pixel_coords) < rad
        self.image = numpy.ma.masked_where(to_mask[None, ...], self.image)

    def plot(self, energy_bin_id=0, unit='deg', **kwargs):
        pyplot.xlabel(f'X [{unit}]')
        pyplot.ylabel(f'Y [{unit}]')
        pyplot.pcolormesh(
            self.xedges.to(unit).value,
            self.yedges.to(unit).value,
            self.image[energy_bin_id].transpose()
        )
        pyplot.colorbar(label='counts')


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
