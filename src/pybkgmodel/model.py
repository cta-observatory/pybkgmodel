import numpy
import os
import astropy.units as u

from astropy.coordinates import SkyCoord

from pybkgmodel.data import MagicEventFile, LstEventFile
from pybkgmodel.data import RunSummary, find_run_neighbours

from pybkgmodel.camera import RectangularCameraImage

class BaseMap:
    
    def __init__(self):
        pass

    def get_runwise_bkg(self):
        pass

    def check_data_format(self, target_run, neighbours):
        if MagicEventFile.is_compatible(target_run.file_name):
                evtfiles = [
                    MagicEventFile(run.file_name, cuts=self.cuts)
                    for run in (target_run,) + neighbours
                ]
                return evtfiles
        elif LstEventFile.is_compatible(target_run.file_name):
                evtfiles = [
                    LstEventFile(run.file_name, cuts=self.cuts)
                    for run in (target_run,) + neighbours
                ]
                return evtfiles
        else:
            raise RuntimeError(f"Unsupported file format for '{target_run.file_name}'.")
        
class WobbleMap:
    """
    Class
    
    Parameters
    ----------
    runs : _type_
        _description_
    x_edges : _type_
        _description_
    y_edges : _type_
        _description_
    e_edges : _type_
        _description_
    cuts : _type_
        _description_
    time_delta : _type_, optional
        _description_, by default 0.2*u.hr
    pointing_delta : _type_, optional
        _description_, by default 2*u.deg
    """
    
    def __init__(self, runs, x_edges, y_edges, e_edges, cuts, time_delta=0.2*u.hr, pointing_delta=2*u.deg):
        """_summary_

        Parameters
        ----------
        runs : _type_
            _description_
        x_edges : _type_
            _description_
        y_edges : _type_
            _description_
        e_edges : _type_
            _description_
        cuts : _type_
            _description_
        time_delta : _type_, optional
            _description_, by default 0.2*u.hr
        pointing_delta : _type_, optional
            _description_, by default 2*u.deg
        """
        self.runs           = runs
        self.xedges         = x_edges
        self.yedges         = y_edges
        self.energy_edges   = e_edges
        self.cuts           = cuts
        self.time_delta     = time_delta
        self.pointing_delta = pointing_delta
        
    def get_runwise_bkg(self, target_run)->RectangularCameraImage:
        """_summary_

        Parameters
        ----------
        target_run : _type_
            _description_

        Returns
        -------
        RectangularCameraImage
            _description_
        """
        neighbours = find_run_neighbours(target_run, self.runs, self.time_delta, self.pointing_delta)

        evtfiles = self.check_data_format(target_run = target_run, neighbours = neighbours)

        images = [
            RectangularCameraImage.from_events(event_file, self.xedges, self.yedges, self.energy_edges)
            for event_file in evtfiles
        ]

        pointing_ra = u.Quantity([event_file.pointing_ra.mean() for event_file in evtfiles])
        pointing_dec =  u.Quantity([event_file.pointing_dec.mean() for event_file in evtfiles])

        src_coord = SkyCoord(
            ra=pointing_ra.mean(),
            dec=pointing_dec.mean()
        )

        for image in images:
            src_cam = src_coord.transform_to(image.center.skyoffset_frame())
            image.mask_half(src_cam)

        counts = numpy.sum([im.counts for im in images], axis=0)
        exposure = u.Quantity([im.exposure for im in images]).sum(axis=0)

        return RectangularCameraImage(counts, self.xedges, self.yedges, self.energy_edges, exposure=exposure)
    
class ExclusionMap:
    def __init__(self, runs, x_edges, y_edges, e_edges, regions, cuts, time_delta=0.2*u.hr, pointing_delta=2*u.deg):
        """_summary_

        Parameters
        ----------
        runs : _type_
            _description_
        x_edges : _type_
            _description_
        y_edges : _type_
            _description_
        e_edges : _type_
            _description_
        regions : _type_
            _description_
        cuts : _type_
            _description_
        time_delta : _type_, optional
            _description_, by default 0.2*u.hr
        pointing_delta : _type_, optional
            _description_, by default 2*u.deg
        """
        self.runs           = runs
        self.xedges         = x_edges
        self.yedges         = y_edges
        self.energy_edges   = e_edges
        self.regions        = regions
        self.cuts           = cuts
        self.time_delta     = time_delta
        self.pointing_delta = pointing_delta

    def get_runwise_bkg(self, target_run)->RectangularCameraImage:
        """_summary_

        Parameters
        ----------
        target_run : _type_
            _description_

        Returns
        -------
        RectangularCameraImage
            _description_
        """
        neighbours = find_run_neighbours(target_run, self.runs, self.time_delta, self.pointing_delta)

        evtfiles = self.check_data_format(target_run = target_run, neighbours = neighbours)

        images = [
            RectangularCameraImage.from_events(event_file, self.xedges, self.yedges, self.energy_edges)
            for event_file in evtfiles
        ]

        for image in images:
            for region in self.regions:
                image.mask_region(region[0])

        counts = numpy.sum([im.counts for im in images], axis=0)
        exposure = u.Quantity([im.exposure for im in images]).sum(axis=0)

        return RectangularCameraImage(counts, self.xedges, self.yedges, self.energy_edges, exposure=exposure)