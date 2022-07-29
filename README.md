# pybkgmodel

## Description
Background model generation tool for Imaging Atmospheric Cherenkov Telescopes (IACTs). Constructs background from the available data excluding the region of interest from the process. 

Supported background generation methods: 

 - "wobble map" - assumes IACT observations were performed wobbling around the target position. For each telescope pointing, the background is generated from the IACT camera half, that does not include the source position;
 - "exclusion map" - excludes the specified sky region from consideration and generates the background model from the remaining data.
 
Supported background generation modes:

 - "run-wise": for each telescope "data run" (observation session unit) identifies other runs close to it in time and constructs the individual background model from them only;
 - "stacked": add the "run-wise" models together, resulting in an observation-averaged background model. The latter is in general less noisy than the individual "run-wise" models at the cost of loosing information on the potential background variation during the observations.

## Installation
Clone and install with `pip`:

```
git clone https://gitlab.pic.es/mstrzys/pybkgmodel.git
cd pybkgmodel
pip install .
```

## Usage
The background model generation is controlled via a configuration file in the YAML format (an example may be found in the "examples" folder). It specifies the input data, output folder, background model generation method, maps binning and exclusion regions to apply.

Execute `bkgmodel` to run the code, specifying the corresponding configuration file, e.g.:

```
bkgmodel --config examples/config_example.yaml
```


## Support
Please use [issues](https://gitlab.pic.es/mstrzys/pybkgmodel/issues) to report problems or make suggestions.

## Roadmap
Despite the initial focus on CTA/LST and MAGIC data, the project may be extended to any other IACTs (e.g. other CTA instruments).

## Contributing
Contributions are welcome.

## Authors and acknowledgment
Original developers are Marcel Strzys, Ievgen Vovk and Moritz Huetten.

## License
We're using GNU GPLv3 here.

## Project status
Active development, so major changes are possible without a notice.
