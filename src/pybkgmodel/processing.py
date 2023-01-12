from functools import reduce
import glob
import inspect
import numpy
from operator import getitem
import os
from regions import Regions
import sys
try:
    import progressbar
except:
    print('Please install the progressbar2 module (not progressbar)')
    sys.exit()
import astropy.units as u

from pybkgmodel.data import RunSummary
from pybkgmodel.model import _WobbleMap, _ExclusionMap
from pybkgmodel.camera import RectangularCameraImage

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

class _BkgMakerBase:
    """ 
    A class used to store the settings from the configuation file and to 
    facilitate the generation of background maps.
    
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
    excl_region : str
        Region to be excluded from the bkg model in ds9 region format.
    bkg_maps : dict 
        Dictionary containing the generated bkg maps and output names for each
        run.
    """    
    
    def __init__(
                self, 
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
                ) -> None:  
        
        """
        Function initializing a prozessing object.
        
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
        time_delta : astropy.units.quantity.Quantity
            Time difference between runs for the run matching. 
        pointing_delta : astropy.units.quantity.Quantity
            Pointing difference between runs for run matching.   
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
        excl_region : str
            Region to be excluded from the bkg model in ds9 region format.
            
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
                                       
        self.time_delta     = time_delta
        self.pointing_delta = pointing_delta
        
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
        
        self.bkg_maps   = {}   
        
        # self._bkg_reco_method = None

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
        
        class_params = inspect.signature(cls).parameters
        
        params_for_init = {}
        
        for current_par in class_params:
            
            try:
                current_par_val = reduce(
                    getitem, 
                    config_class_map["{}".format(current_par)], 
                    config
                    )
            
                if current_par in quantity_list:
                    current_par_val = u.Quantity(current_par_val)
                else:
                    pass
            
            except KeyError:
                print(
                    "Parameter {} missing in config file.".format(config_class_map["{}".format(current_par)])
                    )
                
            params_for_init["{}".format(current_par)] = current_par_val
        
        return cls(**params_for_init)
        
        # return cls(
        #             files          = glob.glob(config['data']['mask']),

        #             cuts           = config['data']['cuts'],
        
        #             out_dir        = config['output']['directory'],
        #             out_prefix     = config['output']['prefix'],
        #             overwrite      = config['output']['overwrite'],           
                    
        #             time_delta     = u.Quantity(config['run_matching']\
        #                                               ['time_delta']),
        #             pointing_delta = u.Quantity(config['run_matching']\
        #                                               ['pointing_delta']),   
        
        #             x_min          = u.Quantity(config['binning']['x']['min']),
        #             x_max          = u.Quantity(config['binning']['x']['max']),
        #             y_min          = u.Quantity(config['binning']['y']['min']),
        #             y_max          = u.Quantity(config['binning']['y']['max']),
        #             x_nbins        = config['binning']['x']['nbins'],
        #             y_nbins        = config['binning']['y']['nbins'],

        #             e_min          = u.Quantity(config['binning']['energy']\
        #                                               ['min']),
        #             e_max          = u.Quantity(config['binning']['energy']\
        #                                               ['max']),
        #             e_nbins        = config['binning']['energy']['nbins'],
      
        #             excl_region    = config['exclusion_regions']
        # )

    def _generate_runwise_maps(self) -> dict:
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
            for ri, run in enumerate(self.runs):
                
                bkg_map = self._bkg_reco_method.get_runwise_bkg(target_run = run)
                
                base_name = os.path.basename(run.file_name)
                base_name, _ = os.path.splitext(base_name)
                
                output_name = os.path.join(
                                        self.out_dir,
                                        f"{self.out_prefix}{base_name}.fits"
                                        )
                
                maps['{}'.format(output_name)] = bkg_map
                  
                progress.update(ri)
        
        self.bkg_maps = maps
        
        return maps
    
    def _stack_maps(self) -> RectangularCameraImage:
        """
        Returns a stacked bkg map of all the runs.

        Returns
        -------
        RectangularCameraImage
        """        
        counts = numpy.sum([m.counts for m in self.bkg_maps.values()],
                           axis=0)
        exposure = u.Quantity([m.exposure for m in self.bkg_maps.values()]
                              ).sum(axis=0)
        stacked_map = RectangularCameraImage(counts, 
                                             self.x_edges, 
                                             self.y_edges, 
                                             self.e_edges, 
                                             exposure=exposure)
        return stacked_map
        
    def _write_maps(self) -> None:
        """
        This method writes the generated bkgmaps to the corresponding output
        path. The output follows the definition in 
        https://gamma-astro-data-formats.readthedocs.io/
        """        
                                           
        for key in self.bkg_maps.keys():
            self.bkg_maps[key].to_hdu().writeto(key, overwrite=self.overwrite)    

class _Runwise(_BkgMakerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_maps(self):
        self._generate_runwise_maps()
        self._write_maps()
        return self.bkg_maps

class _Stacked(_BkgMakerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_maps(self):
        maps = self._generate_runwise_maps()
        stacked_map = self._stack_maps()
        
        stacked_name = os.path.join(
                self.out_dir,
                f"{self.out_prefix}stacked_bkg_map.fits"
                )   
        self.bkg_maps = {stacked_name: stacked_map}
        self._write_maps()
        return self.bkg_maps
    
class RunwiseWobbleMap(_Runwise):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bkg_reco_method = _WobbleMap(runs=self.runs,
                                           x_edges=self.x_edges,
                                           y_edges=self.y_edges,
                                           e_edges=self.e_edges,
                                           cuts=self.cuts,
                                           time_delta=self.time_delta,
                                           pointing_delta=self.pointing_delta
                                           )
        

class StackedWobbleMap(_Stacked):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bkg_reco_method = _WobbleMap(runs=self.runs,
                                           x_edges=self.x_edges,
                                           y_edges=self.y_edges,
                                           e_edges=self.e_edges,
                                           cuts=self.cuts,
                                           time_delta=self.time_delta,
                                           pointing_delta=self.pointing_delta
                                           )

class RunwiseExclusionMap(_Runwise):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.excl_region    = [Regions.parse(reg,format='ds9') for reg in 
                               excl_region]
        self._bkg_reco_method = _ExclusionMap(runs=self.runs,
                                              x_edges=self.x_edges,
                                              y_edges=self.y_edges,
                                              e_edges=self.e_edges,
                                              regions=self.excl_region,
                                              cuts=self.cuts,
                                              time_delta=self.time_delta,
                                              pointing_delta=self.pointing_delta
                                              )

class StackedExclusionMap(_Stacked):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.excl_region    = [Regions.parse(reg,format='ds9') for reg in 
                               excl_region]
        self._bkg_reco_method = _ExclusionMap(runs=self.runs,
                                              x_edges=self.x_edges,
                                              y_edges=self.y_edges,
                                              e_edges=self.e_edges,
                                              regions=self.excl_region,
                                              cuts=self.cuts,
                                              time_delta=self.time_delta,
                                              pointing_delta=self.pointing_delta
                                              )
