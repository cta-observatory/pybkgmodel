import numpy
import os
import astropy.units as u

from astropy.coordinates import SkyCoord

from pybkgmodel.data import MagicEventFile, LstEventFile
from pybkgmodel.data import RunSummary, find_run_neighbours

from pybkgmodel.camera import RectangularCameraImage


def runwise_wobble_map(target_run, runs, xedges, yedges, energy_edges, cuts='None', time_delta=0.2*u.hr, pointing_delta=2*u.deg):
    neighbours = find_run_neighbours(target_run, runs, time_delta, pointing_delta)
    
    if MagicEventFile.is_compatible(target_run.file_name):
            evtfiles = [
                MagicEventFile(run.file_name, cuts=cuts)
                for run in (target_run,) + neighbours
            ]
    elif LstEventFile.is_compatible(target_run.file_name):
            evtfiles = [
                LstEventFile(run.file_name, cuts=cuts)
                for run in (target_run,) + neighbours
            ]
    else:
        raise RuntimeError(f"Unsupported file format for '{target_run.file_name}'.")
    
    images = [
        RectangularCameraImage.from_events(event_file, xedges, yedges, energy_edges)
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

    return RectangularCameraImage(counts, xedges, yedges, energy_edges, exposure=exposure)

def runwise_exclusion_map(target_run, runs, xedges, yedges, energy_edges, regions, cuts='None', time_delta=0.2*u.hr, pointing_delta=2*u.deg):
    neighbours = find_run_neighbours(target_run, runs, time_delta, pointing_delta)
    
    if MagicEventFile.is_compatible(target_run.file_name):
            evtfiles = [
                MagicEventFile(run.file_name, cuts=cuts)
                for run in (target_run,) + neighbours
            ]
    elif LstEventFile.is_compatible(target_run.file_name):
            evtfiles = [
                LstEventFile(run.file_name, cuts=cuts)
                for run in (target_run,) + neighbours
            ]
    else:
        raise RuntimeError(f"Unsupported file format for '{target_run.file_name}'.")
    
    images = [
        RectangularCameraImage.from_events(event_file, xedges, yedges, energy_edges)
        for event_file in evtfiles
    ]

    pointing_ra = u.Quantity([event_file.pointing_ra.mean() for event_file in evtfiles])
    pointing_dec =  u.Quantity([event_file.pointing_dec.mean() for event_file in evtfiles])

    src_coord = SkyCoord(
        ra=pointing_ra.mean(),
        dec=pointing_dec.mean()
    )

    for image in images:
        for region in regions:
            image.mask_region(region[0])

    counts = numpy.sum([im.counts for im in images], axis=0)
    exposure = u.Quantity([im.exposure for im in images]).sum(axis=0)

    return RectangularCameraImage(counts, xedges, yedges, energy_edges, exposure=exposure)