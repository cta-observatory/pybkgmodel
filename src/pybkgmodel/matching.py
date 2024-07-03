import astropy.units as u


class BaseFinder:

    def __init__(self, runs):
        """
        list/tuple of data.RunSummary
        """
        self.runs = runs

    def find_runs(self, target_run):
        pass


class LazyFinder(BaseFinder):

    def find_runs(self, target_run):
        return (target_run,)


class SimpleFinder(BaseFinder):

    def __init__(self, runs, time_delta, pointing_delta):
        super().__init__(runs)
        self.time_delta = time_delta
        self.pointing_delta = pointing_delta

    def find_runs(self, target_run):
        return (target_run,) + self.find_run_neighbours(target_run, self.runs, self.time_delta, self.pointing_delta)

    @classmethod
    def find_run_neighbours(cls, target_run, run_list, time_delta, pointing_delta):

        """
        Returns the neighbours of the specified run.

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
            lambda run_: (abs(run_.mjd_start - target_run.mjd_stop)*u.d < time_delta) or
                        (abs(run_.mjd_stop - target_run.mjd_start)*u.d < time_delta),
            run_list
        )

        neihbours = filter(
            lambda run_: target_run.tel_pointing_start.icrs.separation(run_.tel_pointing_start.icrs)
                        < pointing_delta,
            neihbours
        )

        return tuple(neihbours)
    
class HolizontallyClosestFinder(BaseFinder):
    pass

class RectangleFinder(BaseFinder):
        
    def __init__(self, runs, time_delta, delta_zd, delta_az):
        super().__init__(runs)
        self.time_delta = time_delta
        self.delta_zd = delta_zd
        self.delta_az = delta_az

    def find_runs(self, target_run):
        return (target_run,) + self.find_run_neighbours(target_run, self.runs, self.time_delta, self.delta_zd, self.delta_az)

    @classmethod
    def find_run_neighbours(cls, target_run, run_list, time_delta, delta_zd, delta_az):

        """
        Returns the neighbours of the specified run.

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
        delta_zd: astropy.units.quantity.Quantity
            maximal zenith angle difference from the target run
        delta_az: astropy.units.quantity.Quantity
            maximal azimuth angle difference from the target run

        Developers Note
        ----------
        For simplicity at the moment, the run selection based on alt/az is done
        by the central pointing position, instead of the start/end of runs.
        To more strictly apply the alt/az condition, it may be better to truncate runs
        if a part of run go beyond the alt/az selection window.
        """

        neihbours = filter(
            lambda run_: (abs(run_.mjd_start - target_run.mjd_stop)*u.d < time_delta) or
                        (abs(run_.mjd_stop - target_run.mjd_start)*u.d < time_delta),
            run_list
        )

        neihbours = filter(
            lambda run_: (
                (
                    delta_zd**2 > (
                        (target_run.tel_pointing_start.alt + target_run.tel_pointing_stop.alt)/2 
                        - (run_.tel_pointing_start.alt + run_.tel_pointing_stop.alt)/2
                    )**2
                ) & (
                    delta_az**2 > (
                        (target_run.tel_pointing_start.az + target_run.tel_pointing_stop.az)/2 
                        - (run_.tel_pointing_start.az + run_.tel_pointing_stop.az)/2
                    )**2
                )
            ), neihbours
        )

        return tuple(neihbours)

class CircleFinder(BaseFinder):
    pass