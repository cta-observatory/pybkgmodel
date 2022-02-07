import os
import glob
import numpy
import progressbar
import astropy.units as u

from aux.data import RunSummary
from aux.model import runwise_wobble_map


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

            bkg_map.to_hdu().writeto(
                os.path.join(
                    config['output']['directory'],
                    f"{config['output']['prefix']}{base_name}.fits"
                )
            )

            progress.update(ri)
