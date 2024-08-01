import numpy as np
import astropy.units as u

from astropy.coordinates import SkyCoord

from pybkgmodel.data import MagicEventFile, LstEventFile
from pybkgmodel.data import find_run_neighbours

from pybkgmodel.camera import RectangularCameraImage

__all__ = ["BaseMap", "WobbleMap", "ExclusionMap"]


class BaseMap:

    """
    Base class for the runwise background map reconstruction methods.
    Not intended to be used directly, bust just defines some common
    procedures for all background methods.
    """

    def __init__(self):
        pass

    def read_runs(self, target_run, neighbours, cuts):
        """Function loading the event from the run file into event file objects.

        Parameters
        ----------
        target_run : str
            Run for which the file shall be read.
        neighbours : tuple
            Neighbouring runs selected.
        cuts : str
            Event selection cuts.

        Returns
        -------
        evtfiles : list
            List of the event files objects is returned.

        Raises
        ------
        RuntimeError
            Raise if a run in an unsupported format is provided.
            Currenttly supported formats are DL2 for LST and ROOT for MAGIC.
        """
        if MagicEventFile.is_compatible(target_run.file_name):
            evtfiles = [
            MagicEventFile(run.file_name, cuts=cuts)
            for run in (target_run,) + neighbours
            ]
            return evtfiles
        elif LstEventFile.is_compatible(target_run.file_name):
            evtfiles = [
            LstEventFile(run.file_name, cuts=cuts)
            for run in (target_run,) + neighbours
            ]
            return evtfiles
        else:
            raise RuntimeError(f"Unsupported file format for '{target_run.file_name}'.")


class WobbleMap(BaseMap):
    """
    Class for generating runwise background maps using the wobble map algorithm.

    Attributes
    ----------
    runs : tuple
        Source data.
    x_edges : np.ndarray
        Array of the bin edges along the x/azimuth axis; linear binning.
    y_edges : np.ndarray
        Array of the bin edges along the y/Zenith axis; linear binning.
    e_edges : np.ndarray
        Array of the bin edges in energy; logarithmic binning.
    cuts : str
        Event selection cuts.
    time_delta : astropy.units.quantity.Quantity
        Time difference between runs for the run matching, by default 0.2*u.hr.
    pointing_delta : astropy.units.quantity.Quantity
        Pointing difference between runs for run matching, by default 2*u.deg.
    """

    def __init__(self,
                 runs,
                 x_edges,
                 y_edges,
                 e_edges,
                 cuts,
                 time_delta=0.2*u.hr,
                 pointing_delta=2*u.deg
                 ):
        """
        Function initializing a class for generating runwise background maps using
        the wobble map algorithm.


        Parameters
        ----------
        runs : tuple
            Source data.
        x_edges : np.ndarray
            Array of the bin edges along the x/azimuth axis; linear binning.
        y_edges : np.ndarray
            Array of the bin edges along the y/Zenith axis; linear binning.
        e_edges : np.ndarray
            Array of the bin edges in energy; logarithmic binning.
        cuts : str
            Event selection cuts.
        time_delta : astropy.units.quantity.Quantity
            Time difference between runs for the run matching, by default 0.2*u.hr.
        pointing_delta : astropy.units.quantity.Quantity
            Pointing difference between runs for run matching, by default 2*u.deg.
        """
        self.runs           = runs
        self.xedges         = x_edges
        self.yedges         = y_edges
        self.energy_edges   = e_edges
        self.cuts           = cuts
        self.time_delta     = time_delta
        self.pointing_delta = pointing_delta

    def get_runwise_bkg(self, target_run)->RectangularCameraImage:
        """Function for obtaining runwise background maps using the Wobble
        map method.

        Parameters
        ----------
        target_run : RunSummary
            Run for which the background map shall be generated.

        Returns
        -------
        RectangularCameraImage
            Returns a camera object containing the event counts and exposure
            for each camera bin.
        """
        neighbours = find_run_neighbours(target_run,
                                         self.runs,
                                         self.time_delta,
                                         self.pointing_delta
                                         )

        evtfiles = self.read_runs(target_run = target_run,
                                          neighbours = neighbours,
                                          cuts = self.cuts
                                          )

        images = [
            RectangularCameraImage.from_events(event_file,
                                               self.xedges,
                                               self.yedges,
                                               self.energy_edges
                                               )
            for event_file in evtfiles
        ]

        pointing_ra = u.Quantity([event_file.pointing_ra.mean() for event_file
                                  in evtfiles])
        pointing_dec =  u.Quantity([event_file.pointing_dec.mean() for event_file
                                    in evtfiles])

        src_coord = SkyCoord(
            ra=pointing_ra.mean(),
            dec=pointing_dec.mean()
        )

        for image in images:
            src_cam = src_coord.transform_to(image.center.skyoffset_frame())
            image.mask_half(src_cam)

        counts = np.sum([im.counts for im in images], axis=0)
        exposure = u.Quantity([im.exposure for im in images]).sum(axis=0)

        return RectangularCameraImage(counts, self.xedges,
                                      self.yedges,
                                      self.energy_edges,
                                      exposure=exposure
                                      )

class ExclusionMap(BaseMap):
    """
    Class for generating runwise background maps using the exclusion map
    algorithm.

    Attributes
    ----------
    runs : tuple
        Source data.
    x_edges : np.ndarray
        Array of the bin edges along the x/azimuth axis; linear binning.
    y_edges : np.ndarray
        Array of the bin edges along the y/Zenith axis; linear binning.
    e_edges : np.ndarray
        Array of the bin edges in energy; logarithmic binning.
    excl_region : list
        List of regions to be excluded from the background map in ds9
        format.
    cuts : str
        Event selection cuts.
    time_delta : astropy.units.quantity.Quantity
        Time difference between runs for the run matching, by default 0.2*u.hr.
    pointing_delta : astropy.units.quantity.Quantity
        Pointing difference between runs for run matching, by default 2*u.deg.
    """
    def __init__(self,
                 runs,
                 x_edges,
                 y_edges,
                 e_edges,
                 regions,
                 cuts,
                 time_delta=0.2*u.hr,
                 pointing_delta=2*u.deg
                 ):
        """
        Function initializing a class for generating runwise background maps using
        the exclusion map algorithm.

        Parameters
        ----------
        runs : tuple
            Source data.
        x_edges : np.ndarray
            Array of the bin edges along the x/azimuth axis; linear binning.
        y_edges : np.ndarray
            Array of the bin edges along the y/Zenith axis; linear binning.
        e_edges : np.ndarray
            Array of the bin edges in energy; logarithmic binning.
        excl_region : list
            List of regions to be excluded from the background map in ds9
            format.
        cuts : str
            Event selection cuts.
        time_delta : astropy.units.quantity.Quantity
            Time difference between runs for the run matching, by default 0.2*u.hr.
        pointing_delta : astropy.units.quantity.Quantity
            Pointing difference between runs for run matching, by default 2*u.deg.
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
        """Function for obtaining runwise background maps using the Exclusion
        map method.

        Parameters
        ----------
        target_run : RunSummary
            Run for which the background map shall be generated.

        Returns
        -------
        RectangularCameraImage
            Returns a camera object containing the event counts and exposure
            for each camera bin.
        """
        neighbours = find_run_neighbours(target_run,
                                         self.runs,
                                         self.time_delta,
                                         self.pointing_delta
                                         )

        evtfiles = self.read_runs(target_run = target_run,
                                          neighbours = neighbours,
                                          cuts = self.cuts
                                          )

        images = [
            RectangularCameraImage.from_events(event_file,
                                               self.xedges,
                                               self.yedges,
                                               self.energy_edges
                                               )
            for event_file in evtfiles
        ]

        for image in images:
            for region in self.regions:
                image.mask_region(region[0])

        counts = np.sum([im.counts for im in images], axis=0)
        exposure = u.Quantity([im.exposure for im in images]).sum(axis=0)

        return RectangularCameraImage(counts,
                                      self.xedges,
                                      self.yedges,
                                      self.energy_edges,
                                      exposure=exposure
                                      )
