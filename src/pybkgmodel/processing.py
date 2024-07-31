from functools import reduce
import glob
import inspect
from operator import getitem
import os
import sys

import numpy
from regions import Regions
try:
    import progressbar
except: # pylint: disable=bare-except
    print('Please install the progressbar2 module (not progressbar)')
    sys.exit()
import astropy.units as u

from pybkgmodel.data import RunSummary
from pybkgmodel.model import (WobbleMap,
                              ExclusionMap
                            )
from pybkgmodel.camera import RectangularCameraImage

# list of class attributes, which have a unit assigned
quantity_list = [
                'time_delta',
                'pointing_delta',
                'x_min',
                'x_max',
                'y_min',
                'y_max',
                'e_min',
                'e_max'
                ]

# dictionary to map names in the config file to the class attribute names
config_class_map = {
    'files' : ['data', 'mask'],
    'cuts' : ['data', 'cuts'],
    'out_dir' : ['output', 'directory'],
    'out_prefix' : ['output', 'prefix'],
    'overwrite' : ['output', 'overwrite'],
    'time_delta' : ['run_matching', 'time_delta'],
    'pointing_delta' : ['run_matching', 'pointing_delta'],
    'x_min' : ['binning', 'x', 'min'],
    'x_max' : ['binning', 'x', 'max'],
    'y_min' : ['binning', 'y', 'min'],
    'y_max' : ['binning', 'y', 'max'],
    'x_nbins' : ['binning', 'x', 'nbins'],
    'y_nbins' : ['binning', 'y', 'nbins'],
    'e_min' : ['binning', 'energy', 'min'],
    'e_max' : ['binning', 'energy', 'max'],
    'e_nbins' : ['binning', 'energy', 'nbins'],
    'excl_region' : ['exclusion_regions']
}

class BkgMakerBase:
    """
    Base class for all processing classes, which store the settings from the
    configuation file and facilitate the generation of background maps.
    Not intended for direct usage.

    Attributes
    ----------
    files : list
        List of paths to the files corresponding to the data mask.
    runs : tuple
        Source data.
    cuts : str
        Event selection cuts.
    out_dir : str
        Path where to write the output files to.
    out_prefix : str
        Prefix of the output filename.
    overwrite:  bool
        Whether to overwrite existing output files of same name.
    x_edges : numpy.ndarray
        Array of the bin edges along the x/azimuth axis; linear binning.
    y_edges : numpy.ndarray
        Array of the bin edges along the y/Zenith axis; linear binning.
    e_edges : numpy.ndarray
        Array of the bin edges in energy; logarithmic binning.
    bkg_maps : dict
        Dictionary containing the generated bkg maps and output names for each
        run.
    bkg_map_maker : class
        Class of the background reconstruction algorithm used to obtain the
        runwise background maps.
    """

    def __init__(
                self,
                files,
                cuts,
                out_dir,
                out_prefix,
                overwrite,
                x_min,
                x_max,
                y_min,
                y_max,
                x_nbins,
                y_nbins,
                e_min,
                e_max,
                e_nbins
                ) -> None:

        """
        Function initializing a processing object.

        Parameters
        ----------
        files : list
            List of paths to the files corresponding to the data mask.
        cuts : str
            Event selection cuts.
        out_dir : str
            Path where to write the output files to.
        out_prefix : str
            Prefix of the output filename.
        overwrite:  bool
            Whether to overwrite existing output files of same name.
        x_min : astropy.units.quantity.Quantity
            Minimal positon along the x/azimuth axis.
        x_max : astropy.units.quantity.Quantity
            Maximum positon along the x/azimuth axis.
        x_nbins : int
            Number of bins along the x/azimuth axis.
        y_min : astropy.units.quantity.Quantity
            Minimal positon along the y/Zenith axis.
        y_max : astropy.units.quantity.Quantity
            Maximum positon along the y/Zenith axis.
        y_nbins : int
            Number of bins along the y/Zenith axis.
        e_min : astropy.units.quantity.Quantity
            Minimal energy edge of the bkg maps.
        e_max : astropy.units.quantity.Quantity
            Maximum energy edge of the bkg maps.
        e_nbins : int
            Number of bins along the energy axis

         Returns
        -------
        out
            processing object
        """

        self.files          = glob.glob(files)
        self.runs           = tuple(
                                filter(
                                    lambda r: r.obs_id is not None,
                                    [RunSummary(fname) for fname in
                                    self.files]
                                    )
                                )
        self.cuts           = cuts

        self.out_dir        = out_dir
        self.out_prefix     = out_prefix
        self.overwrite      = overwrite

        self.x_edges        = numpy.linspace(
                                x_min,
                                x_max,
                                x_nbins+1
                                )

        self.y_edges        = numpy.linspace(
                                y_min,
                                y_max,
                                y_nbins+1
                                )

        self.e_edges        = numpy.geomspace(
                                e_min,
                                e_max,
                                e_nbins+1
                                )

        self._bkg_maps   = {}

        self._bkg_map_maker = None


    @property
    def bkg_map_maker(self):
        """Getter for bkg_map_maker."""
        print("This class uses the background method:",
              self.__bkg_map_maker.__class__.__name__)
        return self._bkg_map_maker

    @bkg_map_maker.setter
    def bkg_map_maker(self, value):
        """Setter for bkg_map_maker."""
        self._bkg_map_maker = value

    @property
    def bkg_maps(self):
        """Getter for bkg_maps."""
        return self._bkg_maps

    @classmethod
    def from_config_file(cls, config):
        """
        Function initializing a prozessing object from an input dictionary.

        Parameters
        ----------
        config : dict
            dictionary containing the settings read from the yaml configuration
            file.

        Raises
        ------
        ValueError
            Error is raised if no input dictionary is provided.
        """

        if config is None:
            raise ValueError(
                "No configuration file provided."
            )

        # obtain the parameters of the class on runtime; not know apriori
        class_params = inspect.signature(cls).parameters

        params_for_init = {}

        # Fill dictionary of parameters required by the corresponding class
        # with the values from the config file dictionary
        for current_par in class_params:

            # read required class parameters from the config file dictionary
            try:
                current_par_val = reduce(
                    getitem,
                    config_class_map[f"{current_par}"],
                    config
                    )

                if current_par in quantity_list:
                    current_par_val = u.Quantity(current_par_val)
                else:
                    pass

            except KeyError:
                print(
                    f"Parameter {config_class_map[f'{current_par}']} missing in config file."
                    )

            # assign the extracted parameter to the dictionary from which the
            # class object will be created
            params_for_init[f"{current_par}"] = current_par_val

        return cls(**params_for_init)

    def generate_runwise_maps(self) -> dict:
        """
        Returns a dictionary containing the runwise bkg maps and output file names
        for each input run.

        Returns
        -------
        dict
            {'maps', 'outnames'}
        """
        maps = {}

        with progressbar.ProgressBar(max_value=len(self.runs)) as progress:
            for run_idx, run in enumerate(self.runs):

                # Here the corrsponding bkg reconstruction algorith is applied
                # to obtain the runwise bkg map
                bkg_map = self._bkg_map_maker.get_runwise_bkg(target_run = run)

                # get corresponding names for the bkg maps under which they can
                # be safed
                base_name = os.path.basename(run.file_name)
                base_name, _ = os.path.splitext(base_name)

                output_name = os.path.join(
                                        self.out_dir,
                                        f"{self.out_prefix}{base_name}.fits"
                                        )

                maps[f'{output_name}'] = bkg_map

                progress.update(run_idx)

        self._bkg_maps = maps

        return maps

    @staticmethod
    def stack_maps(bkg_maps, x_edges, y_edges, e_edges) -> RectangularCameraImage:
        """
        Returns a stacked bkg map of all the runs.

        Parameters
        ----------
        bkg_maps : dict
            Dictionary containing the bkg maps and output names for each run.

        x_edges : numpy.ndarray
            Array of the bin edges along the x/azimuth axis; linear binning.

        y_edges : numpy.ndarray
            Array of the bin edges along the y/Zenith axis; linear binning.

        e_edges : numpy.ndarray
            Array of the bin edges in energy; logarithmic binning.

        Returns
        -------
        RectangularCameraImage
        """
        counts = numpy.sum([m.counts for m in bkg_maps.values()],
                           axis=0)
        exposure = u.Quantity([m.exposure for m in bkg_maps.values()]
                              ).sum(axis=0)
        stacked_map = RectangularCameraImage(counts,
                                             x_edges,
                                             y_edges,
                                             e_edges,
                                             exposure=exposure)
        return stacked_map

    @staticmethod
    def write_maps(bkg_maps, overwrite) -> None:
        """
        This method writes the generated bkgmaps to the corresponding output
        path. The output follows the definition in
        https://gamma-astro-data-formats.readthedocs.io/

        Parameters
        ----------
        bkg_maps : dict
            Dictionary containing the bkg maps and output names for each run.

        overwrite:  bool
            Whether to overwrite existing output files of same name.
        """

        for key in bkg_maps.keys():
            bkg_maps[key].to_hdu().writeto(key, overwrite=overwrite)

class Runwise(BkgMakerBase):
    """
    Class defining common functions for the runwise processing classes.
    Not intended for direct usage.
    """

    def get_maps(self):
        """ Method for generating and saving runwise background maps to the
        output file.

        Returns
        -------
        bkg_maps : dict
            Dictionary containing the bkg maps and output names for each run.
        """

        self.generate_runwise_maps()
        self.write_maps(bkg_maps=self.bkg_maps, overwrite=self.overwrite)
        return self.bkg_maps

class Stacked(BkgMakerBase):
    """
    Class defining common functions for the stacked processing classes.
    Not intended for direct usage.
    """

    def get_maps(self):
        """ Method for generating and saving stacked background maps to the
        output file.

        Returns
        -------
        bkg_maps : dict
            Dictionary containing the stacked bkg map and output name.
        """

        self.generate_runwise_maps()
        stacked_map = self.stack_maps(self.bkg_maps,
                                      self.x_edges,
                                      self.y_edges,
                                      self.e_edges
                                      )

        stacked_name = os.path.join(
                self.out_dir,
                f"{self.out_prefix}stacked_bkg_map.fits"
                )
        self._bkg_maps = {stacked_name: stacked_map}
        self.write_maps(bkg_maps=self.bkg_maps, overwrite=self.overwrite)
        return self.bkg_maps

class RunwiseWobbleMap(Runwise):
    """
    A class used to store the settings from the configuation file and to
    facilitate the generation of runwise background maps using the wobble
    background method.

    Attributes
    ----------
    files : list
        List of paths to the files corresponding to the data mask.
    runs : tuple
        Source data.
    cuts : str
        Event selection cuts.
    out_dir : str
        Path where to write the output files to.
    out_prefix : str
        Prefix of the output filename.
    overwrite:  bool
        Whether to overwrite existing output files of same name.
    time_delta : astropy.units.quantity.Quantity
        Time difference between runs for the run matching.
    pointing_delta : astropy.units.quantity.Quantity
        Pointing difference between runs for run matching.
    x_edges : numpy.ndarray
        Array of the bin edges along the x/azimuth axis; linear binning.
    y_edges : numpy.ndarray
        Array of the bin edges along the y/Zenith axis; linear binning.
    e_edges : numpy.ndarray
        Array of the bin edges in energy; logarithmic binning.
    bkg_maps : dict
        Dictionary containing the generated bkg maps and output names for each
        run.
    bkg_map_maker : class
        Class of the background reconstruction algorithm used to obtain the
        runwise background maps.
    """

    def __init__(self,
                files,
                cuts,
                out_dir,
                out_prefix,
                overwrite,
                time_delta,
                pointing_delta,
                x_min,
                x_max,
                y_min,
                y_max,
                x_nbins,
                y_nbins,
                e_min,
                e_max,
                e_nbins
                ):
        """Function initializing a runswise wobble map processing object.

        Parameters
        ----------
        files : list
            List of paths to the files corresponding to the data mask.
        cuts : str
            Event selection cuts.
        out_dir : str
            Path where to write the output files to.
        out_prefix : str
            Prefix of the output filename.
        overwrite:  bool
            Whether to overwrite existing output files of same name.
        x_min : astropy.units.quantity.Quantity
            Minimal positon along the x/azimuth axis.
        x_max : astropy.units.quantity.Quantity
            Maximum positon along the x/azimuth axis.
        x_nbins : int
            Number of bins along the x/azimuth axis.
        y_min : astropy.units.quantity.Quantity
            Minimal positon along the y/Zenith axis.
        y_max : astropy.units.quantity.Quantity
            Maximum positon along the y/Zenith axis.
        y_nbins : int
            Number of bins along the y/Zenith axis.
        e_min : astropy.units.quantity.Quantity
            Minimal energy edge of the bkg maps.
        e_max : astropy.units.quantity.Quantity
            Maximum energy edge of the bkg maps.
        e_nbins : int
            Number of bins along the energy axis
        time_delta : astropy.units.quantity.Quantity
            Time difference between runs for the run matching.
        pointing_delta : astropy.units.quantity.Quantity
            Pointing difference between runs for run matching.
        """

        super().__init__(files,
                        cuts,
                        out_dir,
                        out_prefix,
                        overwrite,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        x_nbins,
                        y_nbins,
                        e_min,
                        e_max,
                        e_nbins
                        )

        self.pointing_delta = pointing_delta
        self.time_delta     = time_delta
        self.bkg_map_maker = WobbleMap(runs=self.runs,
                                        x_edges=self.x_edges,
                                        y_edges=self.y_edges,
                                        e_edges=self.e_edges,
                                        cuts=self.cuts,
                                        time_delta=self.time_delta,
                                        pointing_delta=self.pointing_delta
                                        )

    @property
    def bkg_map_maker(self):
        return super().bkg_map_maker

    @bkg_map_maker.setter
    def bkg_map_maker(self, maker):
        if not isinstance(maker, WobbleMap):
            raise TypeError(f"Maker must be of type {WobbleMap}")
        super(RunwiseWobbleMap, type(self)).bkg_map_maker.__set__(self, maker)

class StackedWobbleMap(Stacked):
    """
    A class used to store the settings from the configuation file and to
    facilitate the generation of a stacked background map using the wobble
    background method.

    Attributes
    ----------
    files : list
        List of paths to the files corresponding to the data mask.
    runs : tuple
        Source data.
    cuts : str
        Event selection cuts.
    out_dir : str
        Path where to write the output files to.
    out_prefix : str
        Prefix of the output filename.
    overwrite:  bool
        Whether to overwrite existing output files of same name.
    time_delta : astropy.units.quantity.Quantity
        Time difference between runs for the run matching.
    pointing_delta : astropy.units.quantity.Quantity
        Pointing difference between runs for run matching.
    x_edges : numpy.ndarray
        Array of the bin edges along the x/azimuth axis; linear binning.
    y_edges : numpy.ndarray
        Array of the bin edges along the y/Zenith axis; linear binning.
    e_edges : numpy.ndarray
        Array of the bin edges in energy; logarithmic binning.
    bkg_maps : dict
        Dictionary containing the generated bkg maps and output names for each
        run.
    bkg_map_maker : class
        Class of the background reconstruction algorithm used to obtain the
        runwise background maps.
    """

    def __init__(self,
                files,
                cuts,
                out_dir,
                out_prefix,
                overwrite,
                time_delta,
                pointing_delta,
                x_min,
                x_max,
                y_min,
                y_max,
                x_nbins,
                y_nbins,
                e_min,
                e_max,
                e_nbins
                ):
        """Function initializing a stacked wobble map processing object.

       Parameters
        ----------
        files : list
            List of paths to the files corresponding to the data mask.
        cuts : str
            Event selection cuts.
        out_dir : str
            Path where to write the output files to.
        out_prefix : str
            Prefix of the output filename.
        overwrite:  bool
            Whether to overwrite existing output files of same name.
        x_min : astropy.units.quantity.Quantity
            Minimal positon along the x/azimuth axis.
        x_max : astropy.units.quantity.Quantity
            Maximum positon along the x/azimuth axis.
        x_nbins : int
            Number of bins along the x/azimuth axis.
        y_min : astropy.units.quantity.Quantity
            Minimal positon along the y/Zenith axis.
        y_max : astropy.units.quantity.Quantity
            Maximum positon along the y/Zenith axis.
        y_nbins : int
            Number of bins along the y/Zenith axis.
        e_min : astropy.units.quantity.Quantity
            Minimal energy edge of the bkg maps.
        e_max : astropy.units.quantity.Quantity
            Maximum energy edge of the bkg maps.
        e_nbins : int
            Number of bins along the energy axis
        time_delta : astropy.units.quantity.Quantity
            Time difference between runs for the run matching.
        pointing_delta : astropy.units.quantity.Quantity
            Pointing difference between runs for run matching.
        """

        super().__init__(files,
                        cuts,
                        out_dir,
                        out_prefix,
                        overwrite,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        x_nbins,
                        y_nbins,
                        e_min,
                        e_max,
                        e_nbins
                        )
        self.pointing_delta = pointing_delta
        self.time_delta     = time_delta
        self.bkg_map_maker  = WobbleMap(runs=self.runs,
                                        x_edges=self.x_edges,
                                        y_edges=self.y_edges,
                                        e_edges=self.e_edges,
                                        cuts=self.cuts,
                                        time_delta=self.time_delta,
                                        pointing_delta=self.pointing_delta
                                        )

    @property
    def bkg_map_maker(self):
        return super().bkg_map_maker

    @bkg_map_maker.setter
    def bkg_map_maker(self, maker):
        if not isinstance(maker, WobbleMap):
            raise TypeError(f"Maker must be of type {WobbleMap}")
        super(StackedWobbleMap, type(self)).bkg_map_maker.__set__(self, maker)

class RunwiseExclusionMap(Runwise):
    """
    A class used to store the settings from the configuation file and to
    facilitate the generation of runwise background maps using the exclusion
    map method.

    Attributes
    ----------
    files : list
        List of paths to the files corresponding to the data mask.
    runs : tuple
        Source data.
    cuts : str
        Event selection cuts.
    out_dir : str
        Path where to write the output files to.
    out_prefix : str
        Prefix of the output filename.
    overwrite:  bool
        Whether to overwrite existing output files of same name.
    time_delta : astropy.units.quantity.Quantity
        Time difference between runs for the run matching.
    pointing_delta : astropy.units.quantity.Quantity
        Pointing difference between runs for run matching.
    x_edges : numpy.ndarray
        Array of the bin edges along the x/azimuth axis; linear binning.
    y_edges : numpy.ndarray
        Array of the bin edges along the y/Zenith axis; linear binning.
    e_edges : numpy.ndarray
        Array of the bin edges in energy; logarithmic binning.
    bkg_maps : dict
        Dictionary containing the generated bkg maps and output names for each
        run.
    bkg_map_maker : class
        Class of the background reconstruction algorithm used to obtain the
        runwise background maps.
    excl_region : list
        List of regions to be excluded from the background map in ds9
        format.
    """

    def __init__(self,
                files,
                cuts,
                out_dir,
                out_prefix,
                overwrite,
                time_delta,
                pointing_delta,
                excl_region,
                x_min,
                x_max,
                y_min,
                y_max,
                x_nbins,
                y_nbins,
                e_min,
                e_max,
                e_nbins
                ):
        """Function initializing a runswise exclusion map processing object.

        Parameters
        ----------
        files : list
            List of paths to the files corresponding to the data mask.
        cuts : str
            Event selection cuts.
        out_dir : str
            Path where to write the output files to.
        out_prefix : str
            Prefix of the output filename.
        overwrite:  bool
            Whether to overwrite existing output files of same name.
        x_min : astropy.units.quantity.Quantity
            Minimal positon along the x/azimuth axis.
        x_max : astropy.units.quantity.Quantity
            Maximum positon along the x/azimuth axis.
        x_nbins : int
            Number of bins along the x/azimuth axis.
        y_min : astropy.units.quantity.Quantity
            Minimal positon along the y/Zenith axis.
        y_max : astropy.units.quantity.Quantity
            Maximum positon along the y/Zenith axis.
        y_nbins : int
            Number of bins along the y/Zenith axis.
        e_min : astropy.units.quantity.Quantity
            Minimal energy edge of the bkg maps.
        e_max : astropy.units.quantity.Quantity
            Maximum energy edge of the bkg maps.
        e_nbins : int
            Number of bins along the energy axis
        time_delta : astropy.units.quantity.Quantity
            Time difference between runs for the run matching.
        pointing_delta : astropy.units.quantity.Quantity
            Pointing difference between runs for run matching.
        excl_region : list
            List of regions to be excluded from the background map in ds9
            format.
        """

        super().__init__(files,
                        cuts,
                        out_dir,
                        out_prefix,
                        overwrite,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        x_nbins,
                        y_nbins,
                        e_min,
                        e_max,
                        e_nbins
                        )
        self.pointing_delta = pointing_delta
        self.time_delta     = time_delta
        self.excl_region    = [Regions.parse(reg,format='ds9') for reg in
                               excl_region]
        self.bkg_map_maker  = ExclusionMap(runs=self.runs,
                                           x_edges=self.x_edges,
                                           y_edges=self.y_edges,
                                           e_edges=self.e_edges,
                                           regions=self.excl_region,
                                           cuts=self.cuts,
                                           time_delta=self.time_delta,
                                           pointing_delta=self.pointing_delta
                                           )

    @property
    def bkg_map_maker(self):
        return super().bkg_map_maker

    @bkg_map_maker.setter
    def bkg_map_maker(self, maker):
        if not isinstance(maker, ExclusionMap):
            raise TypeError(f"Maker must be of type {ExclusionMap}")
        super(RunwiseExclusionMap, type(self)).bkg_map_maker.__set__(self, maker)
        
class StackedExclusionMap(Stacked):
    """
    A class used to store the settings from the configuation file and to
    facilitate the generation of a stacked background map using the exclusion
    map method.

    Attributes
    ----------
    files : list
        List of paths to the files corresponding to the data mask.
    runs : tuple
        Source data.
    cuts : str
        Event selection cuts.
    out_dir : str
        Path where to write the output files to.
    out_prefix : str
        Prefix of the output filename.
    overwrite:  bool
        Whether to overwrite existing output files of same name.
    time_delta : astropy.units.quantity.Quantity
        Time difference between runs for the run matching.
    pointing_delta : astropy.units.quantity.Quantity
        Pointing difference between runs for run matching.
    x_edges : numpy.ndarray
        Array of the bin edges along the x/azimuth axis; linear binning.
    y_edges : numpy.ndarray
        Array of the bin edges along the y/Zenith axis; linear binning.
    e_edges : numpy.ndarray
        Array of the bin edges in energy; logarithmic binning.
    bkg_maps : dict
        Dictionary containing the generated bkg maps and output names for each
        run.
    bkg_map_maker : class
        Class of the background reconstruction algorithm used to obtain the
        runwise background maps.
    excl_region : list
        List of regions to be excluded from the background map in ds9
        format.
    """

    def __init__(self,
                files,
                cuts,
                out_dir,
                out_prefix,
                overwrite,
                time_delta,
                pointing_delta,
                excl_region,
                x_min,
                x_max,
                y_min,
                y_max,
                x_nbins,
                y_nbins,
                e_min,
                e_max,
                e_nbins
                ):
        """Function initializing a stacked exclusion map processing object.

        Parameters
        ----------
        files : list
            List of paths to the files corresponding to the data mask.
        cuts : str
            Event selection cuts.
        out_dir : str
            Path where to write the output files to.
        out_prefix : str
            Prefix of the output filename.
        overwrite:  bool
            Whether to overwrite existing output files of same name.
        x_min : astropy.units.quantity.Quantity
            Minimal positon along the x/azimuth axis.
        x_max : astropy.units.quantity.Quantity
            Maximum positon along the x/azimuth axis.
        x_nbins : int
            Number of bins along the x/azimuth axis.
        y_min : astropy.units.quantity.Quantity
            Minimal positon along the y/Zenith axis.
        y_max : astropy.units.quantity.Quantity
            Maximum positon along the y/Zenith axis.
        y_nbins : int
            Number of bins along the y/Zenith axis.
        e_min : astropy.units.quantity.Quantity
            Minimal energy edge of the bkg maps.
        e_max : astropy.units.quantity.Quantity
            Maximum energy edge of the bkg maps.
        e_nbins : int
            Number of bins along the energy axis
        time_delta : astropy.units.quantity.Quantity
            Time difference between runs for the run matching.
        pointing_delta : astropy.units.quantity.Quantity
            Pointing difference between runs for run matching.
        excl_region : list
            List of regions to be excluded from the background map in ds9
            format.
        """

        super().__init__(files,
                        cuts,
                        out_dir,
                        out_prefix,
                        overwrite,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        x_nbins,
                        y_nbins,
                        e_min,
                        e_max,
                        e_nbins
                        )
        self.pointing_delta = pointing_delta
        self.time_delta     = time_delta
        self.excl_region    = [Regions.parse(reg,format='ds9') for reg in
                               excl_region]
        self.bkg_map_maker  = ExclusionMap(runs=self.runs,
                                           x_edges=self.x_edges,
                                           y_edges=self.y_edges,
                                           e_edges=self.e_edges,
                                           regions=self.excl_region,
                                           cuts=self.cuts,
                                           time_delta=self.time_delta,
                                           pointing_delta=self.pointing_delta
                                           )

    @property
    def bkg_map_maker(self):
        return super().bkg_map_maker

    @bkg_map_maker.setter
    def bkg_map_maker(self, maker):
        if not isinstance(maker, ExclusionMap):
            raise TypeError(f"Maker must be of type {ExclusionMap}")
        super(StackedExclusionMap, type(self)).bkg_map_maker.__set__(self, maker)
