from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
from gammapy.estimators import ExcessMapEstimator
from gammapy.datasets import MapDataset
from gammapy.maps import MapAxis, WcsGeom, Map
from regions import CircleSkyRegion
from astropy import units as u
from astropy.coordinates import Angle
import numpy as np
from gammapy.catalog import CATALOG_REGISTRY
from gammapy.irf import Background2D, Background3D
from gammapy.utils.coordinates import sky_to_fov, fov_to_sky
from .utils import cone_solid_angle, cone_solid_angle_rectangular_pyramid

__all__ = ["ExclusionMapBackgroundMaker"]


class ExclusionMapBackgroundMaker:
    """Exclusion map background algorithm.
    Calculates background in FOV coordinate system aligned with the `ALTAZ` system.

    Parameters
    ----------
    e_reco : `~gammapy.maps.MapAxis`
        Reconstructed energy axis, 
        for example: MapAxis.from_energy_bounds(0.1, 10, 5, unit="TeV", name="energy")
    location : `astropy.coordinates.EarthLocation`
        Location of the telescopes (later the information should be included in the DataStore!).
    nbins : int
        Number of bins to hist the events.
        Default is 20.
    exclusion_radius : str
        Exclusion radius around Fermi sources.
        Default is "0.3 deg".
    offset_max : str
        Maximal offset.
        Default is "1.75 deg"
    """

    def __init__(
        self,
        e_reco,
        location,
        nbins=20,
        exclusion_radius="0.3 deg",
        offset_max="1.75 deg",
    ):
        self.e_reco = e_reco
        self.location = location
        self.nbins = nbins
        self.exclusion_radius = Angle(exclusion_radius)
        self.offset_max = Angle(offset_max)
        self.offset = MapAxis.from_bounds(
            0, self.offset_max, nbin=6, interp="lin", unit="deg", name="offset"
        )
        self.lon_axis = MapAxis.from_bounds(
            -self.offset_max.value,
            self.offset_max.value,
            self.nbins,
            interp="lin",
            unit="deg",
            name="fov_lon",
        )
        self.lat_axis = MapAxis.from_bounds(
            -self.offset_max.value,
            self.offset_max.value,
            self.nbins,
            interp="lin",
            unit="deg",
            name="fov_lat",
        )
        self.counts_map_eff = np.zeros((e_reco.nbin, nbins, nbins))
        self.counts_map_obs = np.zeros((e_reco.nbin, nbins, nbins))
        self.time_map_obs = u.Quantity(np.zeros((nbins, nbins)), u.h)
        self.time_map_eff = u.Quantity(np.zeros((nbins, nbins)), u.h)
        self.get_offset_map()


    def get_offset_map(self):
        """Calculate offset to pointing position for every bin.
        """
        lon, lat = np.meshgrid(self.lon_axis.center.value, self.lat_axis.center.value)
        self.offset_map = np.sqrt(lon ** 2 + lat ** 2)

    def get_exclusion_mask(self, obs):
        """define exclusion mask for all sources in 4fgl catalog in the region"""
        fgl = CATALOG_REGISTRY.get_cls("4fgl")()
        geom = WcsGeom.create(
            skydir=obs.pointing_radec,
            axes=[self.e_reco],
            width= 2 * self.offset_max + 2 * self.exclusion_radius,
            )
        inside_geom = geom.to_image().contains(fgl.positions)
        idx = np.where(inside_geom)[0]
        exclusion_mask = (
            fgl.positions[0].separation(
                obs.events.radec) > self.exclusion_radius
        )
        for id in idx:
            exclusion_mask &= (
                fgl.positions[id].separation(obs.events.radec)
                > self.exclusion_radius
            )
        return exclusion_mask        

    def fill_counts(self, obs, exclusion_mask):
        # hist events in evergy energy bin
        for j in range(self.e_reco.nbin):
            energy_mask = self.e_reco.edges[j] <= obs.events.energy
            energy_mask &= obs.events.energy < self.e_reco.edges[j + 1]
            mask = exclusion_mask & energy_mask
            # convert coordinates from Ra/Dec to Alt/Az
            t = obs.events.time
            frame = AltAz(obstime=t, location=self.location)
            pointing_altaz = obs.events.pointing_radec.transform_to(frame)
            position_events = obs.events.radec.transform_to(frame)
            # convert Alt/Az to Alt/Az FoV
            # effective counts
            lon, lat = sky_to_fov(
                position_events.az[mask],
                position_events.alt[mask],
                pointing_altaz.az[mask],
                pointing_altaz.alt[mask],
            )
            counts_eff, xedges, yedges = np.histogram2d(
                lon.value, lat.value, bins=(
                    self.lon_axis.edges.value, self.lat_axis.edges.value)
            )
            # observed counts
            lon, lat = sky_to_fov(
                position_events.az[energy_mask],
                position_events.alt[energy_mask],
                pointing_altaz.az[energy_mask],
                pointing_altaz.alt[energy_mask],
            )
            counts_obs, xedges, yedges = np.histogram2d(
                lon.value, lat.value, bins=(
                    self.lon_axis.edges.value, self.lat_axis.edges.value)
            )
            #
            if j == 0:
                counts_map_eff = counts_eff
                counts_map_obs = counts_obs
            else:
                counts_map_eff = np.dstack((counts_map_eff, counts_eff))
                counts_map_obs = np.dstack((counts_map_obs, counts_obs))
        counts_map_eff = counts_map_eff.transpose()
        counts_map_obs = counts_map_obs.transpose()
        self.counts_map_eff += counts_map_eff
        self.counts_map_obs += counts_map_obs

    
    def fill_time_maps(self, obs):
        # define exclusion mask for all sources in 4fgl catalog in the region
        fgl = CATALOG_REGISTRY.get_cls("4fgl")()
        geom = WcsGeom.create(
            skydir=obs.pointing_radec,
            axes=[self.e_reco],
            width= 2 * self.offset_max + 2 * self.exclusion_radius,
        )
        inside_geom = geom.to_image().contains(fgl.positions)
        idx = np.where(inside_geom)[0]

        # time_map
        t_binning = np.linspace(obs.tstart.value, obs.tstop.value, 30)
        t_binning = Time(t_binning, format='mjd')
        t_delta = np.diff(t_binning)
        t_center = t_binning[:-1] + 0.5 * t_delta
        lon, lat = np.meshgrid(self.lon_axis.center, self.lat_axis.center)
        # create observation time 2d arrays
        observation_time_obs = np.zeros((self.nbins, self.nbins))
        observation_time_eff = np.zeros((self.nbins, self.nbins))
        # iterate time bins
        pointing_positions = []
        for t_c, t_d in zip(t_center, t_delta):
            # transform from camera coordinates to FoV coordinates (Alt/Az) dependent on the time
            frame = AltAz(obstime=t_c, location=self.location)
            pointing_position = obs.pointing_radec.transform_to(frame)
            pointing_positions.append(pointing_position)
            az, alt = fov_to_sky(
                self.lon_axis.center,
                self.lat_axis.center,
                pointing_position.az,
                pointing_position.alt
            )
            az, alt = np.meshgrid(az, alt)

            coord_radec = SkyCoord(az, alt, frame=frame).transform_to('icrs')
            # calculate masks for FoV and exclusion regions in Ra/Dec coordinates
            coord_lonlat = SkyCoord(lon, lat)
            mask_fov = coord_lonlat.separation(
                SkyCoord(0*u.deg, 0*u.deg)) < self.offset_max
            exclusion_mask = fgl.positions[0].separation(
                coord_radec) > self.exclusion_radius
            for id in idx:
                exclusion_mask &= (
                    fgl.positions[id].separation(
                        coord_radec) > self.exclusion_radius
                )
            # fill observatione time 2d arays
            observation_time_obs[mask_fov] += t_d.to(u.Unit(u.h)).value
            observation_time_eff[exclusion_mask & mask_fov] += t_d.to(u.Unit(u.h)).value
        self.time_map_obs += u.Quantity(observation_time_obs, u.h)
        self.time_map_eff += u.Quantity(observation_time_eff, u.h)
        

    def run(self, data_store, obs_ids=None):
        observations = data_store.get_observations(
            obs_ids,
            required_irf=[])
        for obs in observations:
            exclusion_mask = self.get_exclusion_mask(obs)
            self.fill_counts(obs, exclusion_mask)
            self.fill_time_maps(obs)
        self.alpha_map = self.time_map_eff / self.time_map_obs
        self.bg = self.get_bg_offset(self.counts_map_eff)
        self.bg_rate = self.get_bg_rate()

    def get_bg_offset_1r(self, counts_map):
        rmin = self.offset.edges.value[:-1] * self.offset.unit
        rmax = self.offset.edges.value[1:] * self.offset.unit
        bg_offset = []
        for rmi, rma in zip(rmin, rmax):
            mask = (self.offset_map >= rmi.value) & (
                self.offset_map < rma.value
            )
            sum_counts = np.sum(counts_map[mask])
            solid_angle_diff = cone_solid_angle(rma) - cone_solid_angle(rmi)
            mean_alpha = np.mean(self.alpha_map[mask])
            mean_time = np.mean(self.time_map_obs)
            counts_corrected = sum_counts / mean_alpha / solid_angle_diff / mean_time
            bg_offset.append(counts_corrected.value)
        return np.array(bg_offset) * counts_corrected.unit

    def get_bg_offset(self, counts_map):
        return [self.get_bg_offset_1r(c) for c in self.counts_map_eff]

    def get_bg_rate(self):
        bg_rate = []
        BACKGROUND_UNIT = u.Unit("s-1 MeV-1 sr-1")
        for bg_r, e_width in zip(self.bg, self.e_reco.bin_width):
            a = bg_r / e_width
            a = a.to(BACKGROUND_UNIT)
            bg_rate.append(a)
        return bg_rate

    def get_bg_2d(self):
        BACKGROUND_UNIT = u.Unit("s-1 MeV-1 sr-1")
        bg_2d = Background2D(
            axes=[
                self.e_reco,
                self.offset],
            data=self.bg_rate,
            unit=BACKGROUND_UNIT
        )
        return bg_2d


    def get_bg_3d(self):
        bg_3d_counts = self.counts_map_eff / self.alpha_map
        bg_rate = []
        BACKGROUND_UNIT = u.Unit("s-1 MeV-1 sr-1")
        # calculate solid angle for each pixel
        lon, lat = np.meshgrid(self.lon_axis.bin_width, self.lat_axis.bin_width)
        solid_angle_pixel = cone_solid_angle_rectangular_pyramid(lon, lat)
        # go through every energy bin
        for bg_r, e_width in zip(bg_3d_counts, self.e_reco.bin_width):
            a = bg_r / e_width / self.time_map_obs / solid_angle_pixel
            a[np.isnan(a)] = 0
            a = a.to(BACKGROUND_UNIT)
            bg_rate.append(a)
        bg_3d = Background3D(
            axes = [
                self.e_reco,
                self.lon_axis,
                self.lat_axis],
            data=bg_rate,
            unit=BACKGROUND_UNIT,
            meta = {"FOVALIGN":"ALTAZ"}
        )
        return bg_3d
