from gammapy.data import DataStore
from gammapy.maps import MapAxis
from astropy.coordinates import EarthLocation
from pybkgmodel.exclusion_region_method import ExclusionMapBackgroundMaker
from plotting import (
    plot_alpha_map,
    plot_counts_maps,
    plot_Background2D,
    plot_Background3D,
    plot_Background2D3D,
)


location_hess = EarthLocation(lon=-23.27133, lat=16.5, height=1.8)
data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
e_reco = MapAxis.from_energy_bounds(0.1, 10, 2, unit="TeV", name="energy")
bg = ExclusionMapBackgroundMaker(e_reco, location=location_hess)
bg.run(data_store, [23523, 23526, 23559, 23592])

plot_alpha_map(bg)
plot_counts_maps(bg)
plot_Background2D(bg)
plot_Background3D(bg)
plot_Background2D3D(bg)