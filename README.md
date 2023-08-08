# pybkgmodel [![CI](https://github.com/cta-observatory/pybkgmodel/actions/workflows/ci.yml/badge.svg)](https://github.com/cta-observatory/pybkgmodel/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/cta-observatory/pybkgmodel/branch/main/graph/badge.svg?token=WsJUEfyBsv)](https://codecov.io/gh/cta-observatory/pybkgmodel)
Background Modelling of IACTs


Download the gammapy test datasets using
```
gammapy download datasets
export GAMMAPY_DATA=$PWD/gammapy-datasets
```

Example
```
from pybkgmodel import ExclusionMapBackgroundMaker
from astropy.coordinates import EarthLocation
from astropy import units as u
from gammapy.maps import MapAxis
from gammapy.data import DataStore

longitude = 16.500222 * u.deg
latitude = -23.271778 * u.deg
altitude = 1800 * u.m  
location = EarthLocation(lon=longitude, lat=latitude, height=altitude)

e_reco = MapAxis.from_energy_bounds(
    0.2,
    10,
    7,
    unit="TeV",
    name="energy"
)

bg_maker = ExclusionMapBackgroundMaker(e_reco,location,15)

data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")

obs_id = data_store.obs_table["OBS_ID"][data_store.obs_table["OBJECT"] == "Crab Nebula"]

bg_maker.run(data_store, obs_id)

bg3d = bg_maker.get_bg_3d()

bg2d = bg_maker.get_bg_2d()
```
