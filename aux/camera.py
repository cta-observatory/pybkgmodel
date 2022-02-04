import numpy
from astropy.coordinates import SkyCoord
from matplotlib import pyplot


class CameraImage:
    def __init__(self, image, xedges, yedges, energy_edges):
        self.image = image
        self.xedges = xedges
        self.yedges = yedges
        self.energy_edges = energy_edges
        
    @classmethod
    def from_events(cls, event_file, xedges, yedges, energy_edges):
        image = cls.bin_events(event_file, xedges, yedges, energy_edges)
        
        return CameraImage(image, xedges, yedges, energy_edges)
        
    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'X range':.<20s}: [{self.xedges.min():.1f}, {self.xedges.max():.1f}]
    {'Y range':.<20s}: [{self.yedges.min():.1f}, {self.yedges.max():.1f}]
    {'X bins':.<20s}: {len(self.xedges) - 1}
    {'X bins':.<20s}: {len(self.yedges) - 1}
"""
        )

        return super().__repr__()
        
    @classmethod
    def bin_events(self, event_file, xedges, yedges, energy_edges):
        pass
    
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
    def bin_events(self, event_file, xedges, yedges, energy_edges):
        pointing = SkyCoord(ra=event_file.pointing_ra, dec=event_file.pointing_dec)
        events = SkyCoord(ra=event_file.event_ra, dec=event_file.event_dec)
        
        cam = events.transform_to(pointing.skyoffset_frame())
        
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
