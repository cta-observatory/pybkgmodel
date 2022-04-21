import os
import re
import numpy
import pandas
import uproot
import astropy.time
import astropy.units as u

from astropy.coordinates import SkyCoord, EarthLocation, AltAz


def find_run_neighbours(target_run, run_list, time_delta, pointing_delta):
    """
    Returns the nieghbours of the specified run.

    Parameters
    ----------
    target_run: RunSummary
        Run for which to find the neighbours.
    run_list: iterable
        Runs where to look for the "target_run" neighbours.
    time_delta: astropy.units.quantity.Quantity
        Maximal time difference between either
        (1) the start of the target run and the end of its "neighbour" or
        (2) the end of the target run and the start of its "neighbour"
    pointing_delta: astropy.units.quantity.Quantity
        Maximal pointing difference between the target and the "neibhbour" runs.
    """

    neihbours = filter(
        lambda run_: (abs(run_.mjd_start - target_run.mjd_stop)*u.d < time_delta) or (abs(run_.mjd_stop - target_run.mjd_start)*u.d < time_delta),
        run_list
    )

    neihbours = filter(
        lambda run_: target_run.tel_pointing_start.icrs.separation(run_.tel_pointing_start.icrs) < pointing_delta,
        neihbours
    )

    return tuple(neihbours)


class EventSample:
    def __init__(
            self, 
            event_ra, event_dec, event_energy,
            pointing_ra, pointing_dec, pointing_az, pointing_zd,
            mjd, delta_t
        ):
        self.__event_ra = event_ra
        self.__event_dec = event_dec
        self.__event_energy = event_energy
        self.__pointing_ra = pointing_ra
        self.__pointing_dec = pointing_dec
        self.__pointing_az = pointing_az
        self.__pointing_zd = pointing_zd
        self.__mjd = mjd
        self.__delta_t = delta_t
        self.__eff_obs_time = self.calc_eff_obs_time()
    
    @property
    def delta_t(self):
        return self.__delta_t
        
    @property
    def eff_obs_time(self):
        return self.__eff_obs_time
    
    @property
    def event_ra(self):
        return self.__event_ra
    
    @property
    def event_dec(self):
        return self.__event_dec
    
    @property
    def event_energy(self):
        return self.__event_energy
    
    @property
    def pointing_ra(self):
        return self.__pointing_ra
    
    @property
    def pointing_dec(self):
        return self.__pointing_dec
    
    @property
    def pointing_az(self):
        return self.__pointing_az
    
    @property
    def pointing_zd(self):
        return self.__pointing_zd
    
    @property
    def pointing_alt(self):
        return 90 * u.deg - self.pointing_zd
    
    @property
    def mjd(self):
        return self.__mjd
    
    def calc_eff_obs_time(self):
        mjd_sorted = numpy.sort(self.__mjd)
        time_diff = numpy.diff(mjd_sorted)

        # Dynamic thereshold for the event arrival time difference.
        # Exlcuded the intervals between the runs, that should be
        # a minority if there are > 1000 events in the sample.
        time_diff_max = numpy.percentile(time_diff, 99.9)

        time_diff = time_diff[time_diff < time_diff_max]
        t_elapsed = numpy.sum(time_diff[time_diff < time_diff_max])
        delta_t = self.delta_t[self.delta_t > 0.0 * u.s]

        # Note: though this correction is usually < 1%,
        # this dead time estimate may be inacurate for some instruments.
        dead_time = numpy.amin(delta_t)
        rate = 1 / (numpy.mean(delta_t) - dead_time)

        t_eff = t_elapsed / (1 + rate * dead_time)

        return t_eff


class EventFile:
    file_name = ''
    obs_id = None

    def __init__(self, file_name, cuts=None):
        pass

    def __repr__(self):
        message = f"""{type(self).__name__} instance
    {'File name':.<20s}: {self.file_name}
    {'Obs ID':.<20s}: {self.obs_id}
    {'Alt range':.<20s}: [{self.pointing_alt.min().to(u.deg):.1f}, {self.pointing_alt.max().to(u.deg):.1f}]
    {'Az range':.<20s}: [{self.pointing_az.min().to(u.deg):.1f}, {self.pointing_az.max().to(u.deg):.1f}]
"""
        if self.mjd is not None:
            message += f"    {'MJD range':.<20s}: [{self.mjd.min():.3f}, {self.mjd.max():.3f}]"

        print(message)

        return super().__repr__()

    @classmethod
    def is_compatible(cls, file_name):
        pass

    @classmethod
    def get_obs_id(cls, file_name):
        pass

    @classmethod
    def load_events(cls, file_name, cuts):
        pass

    @property
    def event_ra(self):
        return self.events.event_ra

    @property
    def event_dec(self):
        return self.events.event_dec

    @property
    def event_energy(self):
        return self.events.event_energy

    @property
    def pointing_ra(self):
        return self.events.pointing_ra

    @property
    def pointing_dec(self):
        return self.events.pointing_dec

    @property
    def pointing_az(self):
        return self.events.pointing_az.to(u.deg)

    @property
    def pointing_alt(self):
        return self.events.pointing_alt

    @property
    def mjd(self):
        return self.events.mjd


class MagicEventFile(EventFile):
    def __init__(self, file_name, cuts=None):
        super().__init__(file_name, cuts)

        self.file_name = file_name
        self.obs_id = self.get_obs_id(file_name)
        self.events = self.load_events(file_name, cuts)

    @classmethod
    def is_compatible(cls, file_name):
        _, ext = os.path.splitext(file_name)
        compatible = ext.lower() == ".root"
        return compatible

    @classmethod
    def get_obs_id(cls, file_name):
        parsed = re.findall('.*\d+_(\d+)_\w_[0-9\w]+\-W[\d\.\+]+\.root', file_name)
        if parsed:
            obs_id = int(parsed[0])
        else:
            raise RuntimeError(f'Can not find observations ID in {file_name}')
        
        return obs_id

    @classmethod
    def load_events(cls, file_name, cuts):
        """
        This method loads events from the pre-defiled file and returns them as a dictionary.

        Parameters
        ----------
        file_name: str
            Name of the MAGIC SuperStar/Melibea file to use.

        Returns
        -------
        dict:
            A dictionary with the even properties: charge / arrival time data, trigger, direction etc.
        """

        event_data = dict()

        array_list = [
            #'MTriggerPattern_1.fPrescaled',
            #'MRawEvtHeader_1.fStereoEvtNumber',
            'MRawEvtHeader_1.fDAQEvtNumber',
            'MRawEvtHeader_1.fTimeDiff',
            'MStereoParDisp.fDirectionRA',
            'MStereoParDisp.fDirectionDec',
            'MEnergyEst.fEnergy',
            'MPointingPos_1.fZd',
            'MPointingPos_1.fAz',
            'MPointingPos_1.fRa',
            'MPointingPos_1.fDec',
            'MHadronness.fHadronness'
        ]

        data_units = {
            'event_ra': u.hourangle,
            'event_dec': u.deg,
            'event_energy': u.GeV,
            'pointing_ra':  u.hourangle,
            'pointing_dec': u.deg,
            'pointing_az': u.deg,
            'pointing_zd': u.deg,
            'mjd': u.d,
            'delta_t': u.s,
            'gammaness':u.one
        }

        time_array_list = ['MTime_1.fMjd', 'MTime_1.fTime.fMilliSec', 'MTime_1.fNanoSec']

        mc_array_list = ['MMcEvt_1.fEnergy', 'MMcEvt_1.fTheta', 'MMcEvt_1.fPhi']

        data_names_mapping = {
            #'MTriggerPattern_1.fPrescaled': 'trigger_pattern',
            #'MRawEvtHeader_1.fStereoEvtNumber': 'stereo_event_number',
            'MRawEvtHeader_1.fDAQEvtNumber': 'daq_event_number',
            'MRawEvtHeader_1.fTimeDiff': 'delta_t',
            'MStereoParDisp.fDirectionRA': 'event_ra',
            'MStereoParDisp.fDirectionDec': 'event_dec',
            'MEnergyEst.fEnergy': 'event_energy',
            'MPointingPos_1.fZd': 'pointing_zd',
            'MPointingPos_1.fAz': 'pointing_az',
            'MPointingPos_1.fRa': 'pointing_ra',
            'MPointingPos_1.fDec': 'pointing_dec',
        }

        mc_names_mapping = {
            'MMcEvt_1.fEnergy': 'true_energy',
            'MMcEvt_1.fTheta': 'true_zd',
            'MMcEvt_1.fPhi': 'true_az'
        }

        with uproot.open(file_name) as input_file:
            if 'Events' in input_file:
                data = input_file['Events'].arrays(array_list, cut=cuts, library="np")

                # Mapping the read structure to the alternative names
                for key in data_names_mapping:
                    name = data_names_mapping[key]
                    event_data[name] = data[key]
                    
                event_data['gammaness'] = 1 - data['MHadronness.fHadronness']

                is_mc = 'MMcEvt_1.' in input_file['Events']
                if is_mc:
                    data = input_file['Events'].arrays(mc_array_list, cut=cuts, library="np")

                    # Mapping the read structure to the alternative names
                    for key in data:
                        name = mc_names_mapping[key]
                        event_data[name] = data[key]

                    # Post processing
                    event_data['true_zd'] = numpy.degrees(event_data['true_zd'])
                    event_data['true_az'] = numpy.degrees(event_data['true_az'])
                    # Transformation from Monte Carlo to usual azimuth
                    event_data['true_az'] = -1 * (event_data['true_az'] - 180 + 7)
                else:
                    # Reading the event arrival time information
                    data = input_file['Events'].arrays(time_array_list, cut=cuts, library="np")

                    # Computing the event arrival time
                    mjd = data['MTime_1.fMjd']
                    millisec = data['MTime_1.fTime.fMilliSec']
                    nanosec = data['MTime_1.fNanoSec']

                    event_data['mjd'] = mjd + (millisec / 1e3 + nanosec / 1e9) / 86400.0

            else:
                # The file is likely corrupted, so return empty arrays
                print("The file is corrupted or is missing the event tree. Empty arrays will be returned.")
                for key in data_names_mapping:
                    name = data_names_mapping[key]
                    event_data[name] = numpy.zeros(0)
                    
        finite = [numpy.isfinite(event_data[key]) for key in event_data]
        all_finite = numpy.prod(finite, axis=0, dtype=bool)
        
        for key in event_data:
            event_data[key] = event_data[key][all_finite]

            if key in data_units:
                event_data[key] = event_data[key] * data_units[key]

        event_sample = EventSample(
            event_data['event_ra'],
            event_data['event_dec'],
            event_data['event_energy'],
            event_data['pointing_ra'],
            event_data['pointing_dec'],
            event_data['pointing_az'],
            event_data['pointing_zd'],
            event_data['mjd'],
            event_data['delta_t']
        )

        return event_sample


class LstEventFile(EventFile):
    def __init__(self, file_name, cuts=None):
        super().__init__(file_name, cuts)

        self.file_name = file_name
        self.obs_id = self.get_obs_id(file_name)
        self.events = self.load_events(file_name, cuts)

    @classmethod
    def is_compatible(cls, file_name):
        _, ext = os.path.splitext(file_name)
        compatible = ext.lower() == ".h5"
        return compatible

    @classmethod
    def get_obs_id(cls, file_name):
        parsed = re.findall('.*dl2_LST-1.Run(\d+).h5', file_name)
        if parsed:
            obs_id = int(parsed[0])
        else:
            raise RuntimeError(f'Can not find observations ID in {file_name}')
        
        return obs_id
    
    @classmethod
    def load_events(cls, file_name, cuts):
        """
        This method loads events from the pre-defiled file and returns them as a dictionary.

        Parameters
        ----------
        file_name: str
            Name of the LST DL2 file to use.
        cuts: str
            Cuts to apply to the returned events.

        Returns
        -------
        dict:
            A dictionary with the even properties: charge / arrival time data, trigger, direction etc.
        """

        data_units = {
            'delta_t': u.s,
            'event_ra': u.hourangle,
            'event_dec': u.deg,
            'event_energy': u.TeV,
            'gammaness': u.one,
            'mjd': u.d,
            'pointing_ra': u.degree,
            'pointing_dec':u.degree,
            'pointing_az': u.radian,
            'pointing_zd': u.radian   
        }

        data_names_mapping = {
            'trigger_type': 'trigger_pattern',
            'event_id': 'daq_event_number',
            'reco_ra': 'event_ra',
            'reco_dec': 'event_dec',
            'gammaness': 'gammaness',
            'reco_energy': 'event_energy',
            'mjd':'mjd',
            'delta_t': 'delta_t',
            'az_tel': 'pointing_az',
            'zd_tel': 'pointing_zd',
            'pointing_ra':'pointing_ra',
            'pointing_dec':'pointing_dec',
            'mc_energy': 'true_energy',
            'mc_alt': 'true_zd',
            'mc_az': 'true_az'
        }

        event_data = {data_names_mapping[key]: None for key in data_names_mapping}

        try:
            data = pandas.read_hdf(file_name,key='dl2/event/telescope/parameters/LST_LSTCam')
            if cuts is not None:
                data = data.query(cuts)

            data['zd_tel'] = numpy.radians(90) - data['alt_tel']

            for key in data_names_mapping:
                name = data_names_mapping[key]
                if key in data:
                    event_data[name] = data[key].to_numpy()

            is_mc = "mc_energy" in data

            if not is_mc:
                event_data['mjd'] = astropy.time.Time(data['trigger_time'].to_numpy(), format='unix').mjd

                lst_time = astropy.time.Time(event_data['mjd'], format='mjd')
                lst_loc = EarthLocation(lat=28.761758*u.deg, lon=-17.890659*u.deg, height=2200*u.m)
                alt_az_frame = AltAz(obstime=lst_time, location=lst_loc)
                coords = SkyCoord(alt=data['alt_tel'].to_numpy()*u.rad, az=data['az_tel'].to_numpy()*u.rad, frame=alt_az_frame).icrs

                event_data['pointing_ra'] = coords.ra.to(data_units['pointing_ra']).value
                event_data['pointing_dec'] = coords.dec.to(data_units['pointing_dec']).value
            
        except:
            # The file is likely corrupted, so return empty arrays
            print("The file is corrupted or is missing the event tree. Empty arrays will be returned.")
            for key in data_names_mapping:
                name = data_names_mapping[key]
                event_data[name] = numpy.zeros(0)
                   
        finite = [numpy.isfinite(event_data[key]) for key in event_data if event_data[key] is not None]
        all_finite = numpy.prod(finite, axis=0, dtype=bool)

        for key in event_data:
            if event_data[key] is not None:
                event_data[key] = event_data[key][all_finite]
                    
                if key in data_units:
                    event_data[key] = event_data[key] * data_units[key]
        
        event_sample = EventSample(
            event_data['event_ra'],
            event_data['event_dec'],
            event_data['event_energy'],
            event_data['pointing_ra'],
            event_data['pointing_dec'],
            event_data['pointing_az'],
            event_data['pointing_zd'],
            event_data['mjd'],
            event_data['delta_t']
        )

        return event_sample


class RunSummary:
    __obs_id = None
    __file_name = None
    __tel_pointing_start = None
    __tel_pointing_stop = None

    def __init__(self, file_name):
        if MagicEventFile.is_compatible(file_name):
            events = MagicEventFile(file_name)
        elif LstEventFile.is_compatible(file_name):
            events = LstEventFile(file_name)
        else:
            raise RuntimeError(f"Unsupported file format for '{file_name}'.")
    
        if len(events.mjd) != 0:
            evt_selection = [events.mjd.argmin(), events.mjd.argmax()]
            time = astropy.time.Time(events.mjd[evt_selection], format='mjd')
            # TODO: make location configurable.
            lst_loc = EarthLocation(lat=28.761758*u.deg, lon=-17.890659*u.deg, height=2200*u.m)
            alt_az_frame = AltAz(obstime=time, location=lst_loc)

            pstart, pstop = SkyCoord(events.pointing_az[evt_selection], events.pointing_alt[evt_selection], frame=alt_az_frame)

            self.__file_name = file_name
            self.__obs_id = events.obs_id
            self.__tel_pointing_start = pstart
            self.__tel_pointing_stop = pstop
    
    def __repr__(self):
        print(
f"""{type(self).__name__} instance
    {'Data file':.<20s}: {self.file_name}
    {'Obs ID':.<20s}: {self.obs_id}
    {'MJD start':.<20s}: {self.mjd_start}
    {'MJD stop':.<20s}: {self.mjd_stop}
    {'Duration':.<20s}: {self.obs_duration}
    {'Pointing':.<20s}: {self.tel_pointing_start.icrs}
"""
        )

        return super().__repr__()

    @property
    def obs_id(self):
        return self.__obs_id

    @property
    def file_name(self):
        return self.__file_name

    @property
    def obs_duration(self):
        duration = (self.mjd_stop - self.mjd_start) * u.day
        return duration.to('s')

    @property
    def mjd_start(self):
        return self.tel_pointing_start.frame.obstime.mjd

    @property
    def mjd_stop(self):
        return self.tel_pointing_stop.frame.obstime.mjd

    @property
    def tel_pointing_start(self):
        return self.__tel_pointing_start

    @property
    def tel_pointing_stop(self):
        return self.__tel_pointing_stop

    def to_qtable(self):
        data = {
            'obs_id': [self.obs_id],
            'mjd_start': [self.mjd_start],
            'mjd_stop': [self.mjd_stop],
            'duration': [self.obs_duration],
            'az_tel_start': [self.tel_pointing_start.az.to('deg')],
            'az_tel_stop': [self.tel_pointing_stop.az.to('deg')],
            'alt_tel_start': [self.tel_pointing_start.alt.to('deg')],
            'alt_tel_stop': [self.tel_pointing_stop.alt.to('deg')],
            'ra_tel': [self.tel_pointing_start.icrs.ra.to('deg')],
            'dec_tel': [self.tel_pointing_start.icrs.ra.to('deg')],
            'file_name': [self.file_name]
        }

        return astropy.table.QTable(data)