#!/usr/bin/env python3
import yaml
import argparse

# import regions

from aux.message import message
from aux.data import RunSummary
from aux.processing import process_runwise_wobble_map, process_stacked_wobble_map, process_runwise_exclusion_map, process_stacked_exclusion_map


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="""
        IACT background generation tool
        """
    )

    arg_parser.add_argument(
        "--config", 
        default="config.yaml",
        help='Configuration file to steer the code execution.'
    )
    parsed_args = arg_parser.parse_args()

    config = yaml.load(open(parsed_args.config, "r"), Loader=yaml.SafeLoader)
    
    supported_modes = (
        'runwise_exclusion',
        'runwise_wobble',
        'stacked_exclusion',
        'stacked_wobble',
    )
    
    if config['mode'] not in supported_modes:
        raise ValueError(f"Unsupported mode '{config['mode']}', valid choices are '{supported_modes}'")

    message(f'Generating background maps')
    if config['mode'] == 'runwise_wobble':
        process_runwise_wobble_map(config)
    elif config['mode'] == 'stacked_wobble':
        process_stacked_wobble_map(config)
    elif config['mode'] == 'runwise_exclusion':
        process_runwise_exclusion_map(config)
    elif config['mode'] == 'stacked_exclusion':
        process_stacked_exclusion_map(config)
    else:
        ValueError(f"Unsupported mode '{config['mode']}'. This should have been caught earlier.")
