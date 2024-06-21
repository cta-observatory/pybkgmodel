import numpy
import astropy.units as u

from astropy.coordinates import SkyCoord

from pybkgmodel.data import load_file
# from pybkgmodel.data import find_run_neighbours

from pybkgmodel.camera import RectangularCameraImage
from pybkgmodel.matching import LazyFinder, SimpleFinder


class BaseMap:

    """
    Base class for the runwise background map reconstruction methods.
    Not intended to be used directly, bust just defines some common
    procedures for all background methods.
    """

    def __init__(self):
        pass

    # def read_runs(self, target_run, neighbours, cuts):
    #     """Function loading the event from the run file into event file objects.

    #     Parameters
    #     ----------
    #     target_run : str
    #         Run for which the file shall be read.
    #     neighbours : tuple
    #         Neighbouring runs selected.
    #     cuts : str
    #         Event selection cuts.

    #     Returns
    #     -------
    #     evtfiles : list
    #         List of the event files objects is returned.

    #     Raises
    #     ------
    #     RuntimeError
    #         Raise if a run in an unsupported format is provided.
    #         Currenttly supported formats are DL3 according to GADF, DL2 for LST and ROOT for MAGIC.
    #     """
    #     if MagicRootEventFile.is_compatible(target_run.file_name):
    #         evtfiles = [
    #         MagicRootEventFile(run.file_name, cuts=cuts)
    #         for run in (target_run,) + neighbours
    #         ]
    #         return evtfiles
    #     elif LstDL2EventFile.is_compatible(target_run.file_name):
    #         evtfiles = [
    #         LstDL2EventFile(run.file_name, cuts=cuts)
    #         for run in (target_run,) + neighbours
    #         ]
    #         return evtfiles
    #     elif DL3EventFile.is_compatible(target_run.file_name):
    #         evtfiles = [
    #         DL3EventFile(run.file_name)
    #         for run in (target_run,) + neighbours
    #         ]
    #         return evtfiles
    #     else:
    #         raise RuntimeError(f"Unsupported file format for '{target_run.file_name}'.")


class Map(BaseMap):

    def __init__(
        self,
        runs,
        x_edges,
        y_edges,
        e_edges,
        cuts,
        time_delta=0.2*u.hr,
        pointing_delta=2*u.deg,
        neighbouring_mode = 'lazy'
    ):
        """
        Function initializing a class for generating runwise background maps using
        the wobble map algorithm.

        Parameters
        ----------
        runs : tuple
            Source data.
        x_edges : numpy.ndarray
            Array of the bin edges along the x/azimuth axis; linear binning.
        y_edges : numpy.ndarray
            Array of the bin edges along the y/Zenith axis; linear binning.
        e_edges : numpy.ndarray
            Array of the bin edges in energy; logarithmic binning.
        cuts : str
            Event selection cuts.
        time_delta : astropy.units.quantity.Quantity
            Time difference between runs for the run matching, by default 0.2*u.hr.
        pointing_delta : astropy.units.quantity.Quantity
            Pointing difference between runs for run matching, by default 2*u.deg.
        """
        self.runs              = runs
        self.xedges            = x_edges
        self.yedges            = y_edges
        self.energy_edges      = e_edges
        self.cuts              = cuts
        self.time_delta        = time_delta
        self.pointing_delta    = pointing_delta
        self.neighbouring_mode = neighbouring_mode


    def get_single_run_map(self,):
        pass


    def find_neighbours(self, target_run):

        # specify the run-matching mode
        if self.neighbouring_mode == 'lazy':
            finder = LazyFinder(runs = self.runs)
        elif self.neighbouring_mode == 'simple':
            finder = SimpleFinder(
                runs = self.runs,
                time_delta = self.time_delta,
                pointing_delta = self.pointing_delta,
            )
        elif self.neighbouring_mode == 'rectangle':
            raise NotImplementedError
        elif self.neighbouring_mode == 'circle':
            raise NotImplementedError
        else:
            raise NotImplementedError
        runs = finder.find_runs(target_run)

        return runs
    

    @staticmethod
    def stack_images(images, *args, **kwargs) -> RectangularCameraImage:

        """
        Stack maps into a sigle map

        Parameters
        ----------
        maps : list of CameraImage


        Returns
        -------
        RectangularCameraImage
        """

        # check the geometry consistency
        # TODO: this functionality should be in the CameraImage class?
        # TODO: check the same CameraImage class or not (e.g. RectangularCameraImage)
        if len(images) < 2:
            pass
        else:
            arrays_x = [im.xedges for im in images]
            arrays_y = [im.yedges for im in images]
            arrays_e = [im.energy_edges for im in images]
            for arrays in [arrays_x, arrays_y, arrays_e]:
                is_aligned = [numpy.allclose(arrays[0], arr) for arr in arrays]
                if not is_aligned:
                    raise ValueError('the camera image geometries are not aligned')
        
        xedges = images[0].xedges
        yedges = images[0].yedges
        energy_edges = images[0].energy_edges
        
        # stack
        counts = numpy.sum([im.counts for im in images], axis=0)
        exposure = numpy.sum(u.Quantity([im.exposure for im in images]), axis=0)

        # ToDo: automatically use the same class with the input images
        stacked_map = RectangularCameraImage(
            counts = counts,
            exposure = exposure,
            xedges = xedges,
            yedges = yedges,
            energy_edges = energy_edges,
            *args, **kwargs,
        )

        return stacked_map


class WobbleMap(Map):
    """
    Class for generating runwise background maps using the wobble map algorithm.

    Attributes
    ----------
    runs : tuple
        Source data.
    x_edges : numpy.ndarray
        Array of the bin edges along the x/azimuth axis; linear binning.
    y_edges : numpy.ndarray
        Array of the bin edges along the y/Zenith axis; linear binning.
    e_edges : numpy.ndarray
        Array of the bin edges in energy; logarithmic binning.
    cuts : str
        Event selection cuts.
    time_delta : astropy.units.quantity.Quantity
        Time difference between runs for the run matching, by default 0.2*u.hr.
    pointing_delta : astropy.units.quantity.Quantity
        Pointing difference between runs for run matching, by default 2*u.deg.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.src_coord = None # TODO: maybe better that users can specify the source in the config file?


    def get_single_run_map(self, evtfile, src_coord) -> RectangularCameraImage:
        
        image = RectangularCameraImage.from_events(
            evtfile,
            self.xedges,
            self.yedges,
            self.energy_edges
        )
        src_cam = src_coord.transform_to(image.center.skyoffset_frame())
        image.mask_half(src_cam)

        return image


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

        # specifies runs (target_run + neighbours)
        runs = self.find_neighbours(target_run)

        # load files
        # TODO: truncate the run trajectory to keep it close to the target run
        evtfiles = [
            load_file(run.file_name, self.cuts)
            for run in runs
        ]

        # guess the position of the source of interest
        # TODO: maybe better that users can specify the source in config?
        pointing_ra = u.Quantity([
            event_file.pointing_ra.mean() for event_file in evtfiles
        ])
        pointing_dec =  u.Quantity([
            event_file.pointing_dec.mean() for event_file in evtfiles
        ])
        src_coord = SkyCoord(
            ra = pointing_ra.mean(),
            dec = pointing_dec.mean()
        )

        # get the wobble maps (half masked)
        images = [self.get_single_run_map(evtfile, src_coord) for evtfile in evtfiles]

        # stack
        image = self.stack_images(images)

        return image
    


class ExclusionMap(Map):
    """
    Class for generating runwise background maps using the exclusion map
    algorithm.

    Attributes
    ----------
    runs : tuple
        Source data.
    x_edges : numpy.ndarray
        Array of the bin edges along the x/azimuth axis; linear binning.
    y_edges : numpy.ndarray
        Array of the bin edges along the y/Zenith axis; linear binning.
    e_edges : numpy.ndarray
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
    
    def __init__(self, regions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regions = regions

    
    def get_single_run_map(self, evtfile) -> RectangularCameraImage:
        
        image = RectangularCameraImage.from_events(
            evtfile,
            self.xedges,
            self.yedges,
            self.energy_edges
        )
        for region in self.regions:
            image.mask_region(region[0])

        return image


    def get_runwise_bkg(self, target_run) -> RectangularCameraImage:
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

        # specifies runs (target_run + neighbours)
        runs = self.find_neighbours(target_run)

        # load files
        # TODO: truncate the run trajectory to keep it close to the target run
        evtfiles = [
            load_file(run.file_name, self.cuts)
            for run in runs
        ]

        # get the excluded images
        images = [self.get_single_run_map(evtfile) for evtfile in evtfiles]

        # stack
        image = self.stack_images(images)

        return image
