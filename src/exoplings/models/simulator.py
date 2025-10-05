import batman
import numpy as np
import swyft

DUR2PER = 1 / 0.0254921


class Simulator(swyft.Simulator):
    def __init__(self, rand_b=False, rand_dur=False, rand_t0=False, t_len=250):
        super().__init__()
        self.rand_b = rand_b
        self.rand_dur = rand_dur
        self.rand_t0 = rand_t0
        self.t_len = t_len
        self.transform_samples = swyft.to_numpy32

    def sample_z(self):
        # rp_sqrt = np.random.uniform(low=-0.15, high=0.5477225575051661)
        # rp = np.heaviside( rp_sqrt, 1.) * rp_sqrt**2

        rp = np.random.uniform(low=0.0, high=0.5477225575051661) ** 2
        # rp = np.random.uniform(low=0.03162277660168379, high=0.5477225575051661)**2
        # rp = np.random.uniform(low=0.1, high=0.16)

        if self.rand_dur:
            dur = np.random.uniform(low=0.025, high=0.075)
        else:
            dur = 0.05

        if self.rand_b:
            b = np.random.uniform(low=0.0, high=1.0)
        else:
            b = 0.0

        if self.rand_t0:
            t0 = np.random.uniform(low=-0.01, high=0.01)
        else:
            t0 = 0.0

        return np.array([rp, b, dur, t0])

    def calc_m(self, z):
        m = self.phys_sim(rp=z[0], b=z[1], dur=z[2], t0=z[3], t_len=self.t_len)
        return m.astype(np.float32)

    def calc_x(self, m):
        sigma = 0.0005
        # sigma = 0.00001
        result = self.get_noisy(m, sigma=sigma)
        return result.astype(np.float32)

    def build(self, graph):  # the print statements are for illustration only
        print("--- Building graph!")
        # z = graph.node('z', lambda: np.random.uniform(low=0., high=0.3))
        # rp, per, inc, t0
        z = graph.node("z", self.sample_z)
        m = graph.node("m", self.calc_m, z)
        x = graph.node("x", self.calc_x, m)

        # # Store event trust scores
        # trust_scores = graph.node('trust', lambda: self.pass_events[-1])

        print("--- x =", x)
        print("--- m =", m)
        print("--- z =", z)

    def phys_sim(self, rp, b=0.0, dur=0.025, t0=0.0, t_len=250):
        params = batman.TransitParams()  # object to store transit parameters
        params.t0 = t0  # time of inferior conjunction
        params.per = DUR2PER * dur  # orbital period
        params.rp = rp  # planet radius (in units of stellar radii)
        params.a = 15.0  # semi-major axis (in units of stellar radii)
        params.inc = np.rad2deg(np.arccos(b / 15.0))  # orbital inclination (in degrees)
        params.ecc = 0.0  # eccentricity
        params.w = 90.0  # longitude of periastron (in degrees)
        params.limb_dark = "uniform"  # limb darkening model
        params.u = []
        # params.u = [0.5, 0.1, 0.1, -0.1]        #limb darkening coefficients [u1, u2, u3, u4]

        t = np.linspace(-0.05, 0.05, t_len)  # times at which to calculate light curve
        m = batman.TransitModel(params, t)  # initializes model

        flux = m.light_curve(params)  # calculates light curve
        return flux

    def get_noisy(self, y, sigma=0.005):
        y_noisy = y + np.random.normal(loc=0.0, scale=sigma, size=len(y))
        return y_noisy
