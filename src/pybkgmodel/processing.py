import os
import glob
import numpy
from regions import Regions
import sys
try:
    import progressbar
except:
    print('Please install the progressbar2 module (not progressbar)')
    sys.exit()
import astropy.units as u

from aux.data import RunSummary
from aux.model import runwise_wobble_map, runwise_exclusion_map
from aux.camera import RectangularCameraImage


def process_runwise_wobble_map(config):
    cuts = config['data']['cuts']

    xedges = numpy.linspace(
        u.Quantity(config['binning']['x']['min']),
        u.Quantity(config['binning']['x']['max']),
        num=config['binning']['x']['n']+1
    )
    yedges = numpy.linspace(
        u.Quantity(config['binning']['y']['min']),
        u.Quantity(config['binning']['y']['max']),
        num=config['binning']['y']['n']+1
    )
    energy_edges = numpy.geomspace(
        u.Quantity(config['binning']['energy']['min']),
        u.Quantity(config['binning']['energy']['max']),
        num=config['binning']['energy']['n']+1
    )

    files = glob.glob(config['data']['mask'])

    runs = [RunSummary(fname) for fname in files]
    runs = tuple(filter(lambda r: r.obs_id is not None, runs))

    with progressbar.ProgressBar(max_value=len(runs)) as progress:
        for ri, run in enumerate(runs):
            
            # TODO: get all params via config
            bkg_map = runwise_wobble_map(
                run,
                runs,
                xedges,
                yedges,
                energy_edges,
                cuts=cuts,
                time_delta=0.2*u.hr,
                pointing_delta=2*u.deg
            )

            base_name = os.path.basename(run.file_name)
            base_name, _ = os.path.splitext(base_name)
            
            output_name = os.path.join(
            config['output']['directory'],
            f"{config['output']['prefix']}{base_name}.fits"
            )
            if not config['output']['overwrite']:
                if os.path.exists(output_name):
                    print('Output file %s already existing. Abort'%output_name)
                    sys.exit()

            bkg_map.to_hdu().writeto(output_name, overwrite=config['output']['overwrite'])

            progress.update(ri)


def process_stacked_wobble_map(config):
    
    output_name = os.path.join(
        config['output']['directory'],
        f"{config['output']['prefix']}stacked_wobble_map.fits"
    )    
    if not config['output']['overwrite']:
        if os.path.exists(output_name):
            print('Output file %s already existing. Abort'%output_name)
            sys.exit()
    
    cuts = config['data']['cuts']

    xedges = numpy.linspace(
        u.Quantity(config['binning']['x']['min']),
        u.Quantity(config['binning']['x']['max']),
        num=config['binning']['x']['n']+1
    )
    yedges = numpy.linspace(
        u.Quantity(config['binning']['y']['min']),
        u.Quantity(config['binning']['y']['max']),
        num=config['binning']['y']['n']+1
    )
    energy_edges = numpy.geomspace(
        u.Quantity(config['binning']['energy']['min']),
        u.Quantity(config['binning']['energy']['max']),
        num=config['binning']['energy']['n']+1
    )

    files = glob.glob(config['data']['mask'])

    runs = [RunSummary(fname) for fname in files]
    runs = tuple(filter(lambda r: r.obs_id is not None, runs))

    with progressbar.ProgressBar(max_value=len(runs)) as progress:
        maps = []
        for ri, run in enumerate(runs):
            # TODO: get all params via config
            map_ = runwise_wobble_map(
                run,
                runs,
                xedges,
                yedges,
                energy_edges,
                cuts=cuts,
                time_delta=0.2*u.hr,
                pointing_delta=2*u.deg
            )

            maps.append(map_)

            progress.update(ri)

    counts = numpy.sum([m.counts for m in maps], axis=0)
    exposure = u.Quantity([m.exposure for m in maps]).sum(axis=0)
    stacked_map = RectangularCameraImage(counts, xedges, yedges, energy_edges, exposure=exposure)

    stacked_map.to_hdu().writeto(output_name, overwrite=config['output']['overwrite'])


def process_runwise_exclusion_map(config):
    cuts = config['data']['cuts']
    
    regions = [Regions.parse(reg,format='ds9') for reg in config['exclusion_regions']]

    xedges = numpy.linspace(
        u.Quantity(config['binning']['x']['min']),
        u.Quantity(config['binning']['x']['max']),
        num=config['binning']['x']['n']+1
    )
    yedges = numpy.linspace(
        u.Quantity(config['binning']['y']['min']),
        u.Quantity(config['binning']['y']['max']),
        num=config['binning']['y']['n']+1
    )
    energy_edges = numpy.geomspace(
        u.Quantity(config['binning']['energy']['min']),
        u.Quantity(config['binning']['energy']['max']),
        num=config['binning']['energy']['n']+1
    )

    files = glob.glob(config['data']['mask'])

    runs = [RunSummary(fname) for fname in files]
    runs = tuple(filter(lambda r: r.obs_id is not None, runs))

    with progressbar.ProgressBar(max_value=len(runs)) as progress:
        for ri, run in enumerate(runs):
            
            # TODO: get all params via config
            bkg_map = runwise_exclusion_map(
                run,
                runs,
                xedges,
                yedges,
                energy_edges,
                regions,
                cuts=cuts,
                time_delta=0.2*u.hr,
                pointing_delta=2*u.deg
            )

            base_name = os.path.basename(run.file_name)
            base_name, _ = os.path.splitext(base_name)
            
            output_name = os.path.join(
            config['output']['directory'],
            f"{config['output']['prefix']}{base_name}.fits"
            )
            if not config['output']['overwrite']:
                if os.path.exists(output_name):
                    print('Output file %s already existing. Abort'%output_name)
                    sys.exit()

            bkg_map.to_hdu().writeto(output_name, overwrite=config['output']['overwrite'])

            progress.update(ri)

def process_stacked_exclusion_map(config):
    
    output_name = os.path.join(
        config['output']['directory'],
        f"{config['output']['prefix']}stacked_exclusion_map.fits"
    )    
    if not config['output']['overwrite']:
        if os.path.exists(output_name):
            print('Output file %s already existing. Abort'%output_name)
            sys.exit()
    
    cuts = config['data']['cuts']
    
    regions = [Regions.parse(reg,format='ds9') for reg in config['exclusion_regions']]

    xedges = numpy.linspace(
        u.Quantity(config['binning']['x']['min']),
        u.Quantity(config['binning']['x']['max']),
        num=config['binning']['x']['n']+1
    )
    yedges = numpy.linspace(
        u.Quantity(config['binning']['y']['min']),
        u.Quantity(config['binning']['y']['max']),
        num=config['binning']['y']['n']+1
    )
    energy_edges = numpy.geomspace(
        u.Quantity(config['binning']['energy']['min']),
        u.Quantity(config['binning']['energy']['max']),
        num=config['binning']['energy']['n']+1
    )

    files = glob.glob(config['data']['mask'])

    runs = [RunSummary(fname) for fname in files]
    runs = tuple(filter(lambda r: r.obs_id is not None, runs))

    with progressbar.ProgressBar(max_value=len(runs)) as progress:
        maps = []
        for ri, run in enumerate(runs):
            # TODO: get all params via config
            map_ = runwise_exclusion_map(
                run,
                runs,
                xedges,
                yedges,
                energy_edges,
                regions,
                cuts=cuts,
                time_delta=0.2*u.hr,
                pointing_delta=2*u.deg
            )

            maps.append(map_)

            progress.update(ri)

    counts = numpy.sum([m.counts for m in maps], axis=0)
    exposure = u.Quantity([m.exposure for m in maps]).sum(axis=0)
    stacked_map = RectangularCameraImage(counts, xedges, yedges, energy_edges, exposure=exposure)

    stacked_map.to_hdu().writeto(output_name, overwrite=config['output']['overwrite'])
