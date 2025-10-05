import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.constants import R_earth, R_sun


class PlanetDetailExtractor:
    def __init__(self, telescope="kepler"):
        self.r_earth = R_earth.value
        self.r_sun = R_sun.value
        self.telescope = telescope
        if telescope == "kepler":
            df_kepl = pd.read_csv("data_csv/kepler.csv", skiprows=53, header=0)
            self.df = df_kepl[df_kepl["koi_disposition"] == "CONFIRMED"]
            #### apply extra filters? 'koi_model_snr','koi_tce_plnt_num'
        elif telescope == "tess":
            df_tess = pd.read_csv("data_csv/tess.csv", skiprows=69, header=0)
            self.df = df_tess[df_tess["tfopwg_disp"] == "KP"]
        else:
            print("Telescope not found.")

    def confirmed_planets(self):
        if self.telescope == "kepler":
            return self.df[["kepid", "kepler_name"]]
        elif self.telescope == "tess":
            return self.df[["toi", "tid"]]

    def convert2convention_kepler(self, localdf):
        z = localdf["koi_prad"].values[0] * self.r_earth / (localdf["koi_srad"].values[0] * self.r_sun)
        localdict = {
            "z": z,
            "t0": localdf["koi_time0bk"].values[0],
            "per": localdf["koi_period"].values[0],
            "impact": localdf["koi_impact"].values[0],
            "duration": localdf["koi_duration"].values[0] / 24.0,
            #  'depth'    : localdf['koi_depth'].values[0]
        }
        return localdict

    def convert2convention_tess(self, localdf):
        # planet-to-star radius ratio
        z = localdf["pl_rade"].values[0] * self.r_earth / (localdf["st_rad"].values[0] * self.r_sun)

        localdict = {
            "z": z,
            "t0": localdf["pl_tranmid"].values[0] - 2457000.0,  # mid-transit time [BTJD]
            "per": localdf["pl_orbper"].values[0],  # orbital period [days]
            "impact": None,  # not in TOI table
            "duration": localdf["pl_trandurh"].values[0] / 24.0,  # hours → days
            # "depth"   : localdf['pl_trandep'].values[0]               # fractional depth
        }
        return localdict

    def find_planet_details_tess(self, planet_name: int):
        # The catalog may use 'toi' or 'pl_name' as identifier; adjust as needed
        # if 'pl_name' in self.df.columns:
        #     localdf = self.df[self.df['pl_name'] == planet_name].copy()
        if "toi" in self.df.columns:
            localdf = self.df[self.df["tid"] == planet_name].copy()
        else:
            print("No matching planet name column in TESS catalog")
            return None

        if len(localdf) == 0:
            print(f"{planet_name} not found.")
            return None
        else:
            return self.convert2convention_tess(localdf)

    def find_planet_details_kepler(self, planet_name: str):
        # df['koi_period']    #days
        # df['koi_impact']    #impact parameter
        # df['koi_duration']  #hrs
        # df['koi_depth']     #ppm
        # df['koi_prad']      #earth radii
        # df['koi_srad']      #solar radii
        # df['koi_model_snr']
        # df['koi_tce_plnt_num']
        # try:
        localdf = self.df[self.df["kepler_name"] == planet_name][
            ["koi_time0bk", "koi_period", "koi_impact", "koi_duration", "koi_depth", "koi_prad", "koi_srad", "koi_model_snr", "koi_tce_plnt_num"]
        ].copy()
        if len(localdf) == 0:
            print("Planet not found.")
            return None
        else:
            return self.convert2convention_kepler(localdf)

    def find_planet_details(self, planet_name: str | int):
        if self.telescope == "kepler":
            return self.find_planet_details_kepler(planet_name)
        elif self.telescope == "tess":
            return self.find_planet_details_tess(planet_name)

    def find_data_kepler(self, planet_name, period_days, t0_btjd, window):
        # --- CONFIGURATION ---
        # target_name = "WASP-18"
        # period_days = 0.94145299   # orbital period from literature
        # t0_bjd = 2454221.48163     # mid-transit time (BJD_TDB) from discovery papers
        # window = 0.3              # days around each transit to extract

        ## Single search
        # look up the KIC ID from your dataframe
        row = self.df[self.df["kepler_name"] == planet_name]
        if len(row) == 0:
            raise ValueError("Planet not found in local catalog")

        kepid = row["kepid"].values[0]
        print(f"Searching Kepler lightcurves for {planet_name} (KIC {kepid}) ...")

        # try:
        #   print(f"Searching for short cadence...")
        #   lc_files = lk.search_lightcurve(f"KIC {kepid}", author="Kepler", cadence="short").download_all()
        #   print(f"Found!")
        # except:
        print("Searching for 2-min cadence...")
        lc_files = lk.search_lightcurve(f"KIC {kepid}", author="Kepler", cadence="short").download_all()
        print("Found!")

        # # --- DOWNLOAD TESS PDCSAP LIGHTCURVE FILES ---
        # print(f"Searching Kepler lightcurves for {planet_name} ...")
        # lc_files = lk.search_lightcurve(planet_name, author="Kepler", cadence="short").download_all()

        # lc_files = search_lightcurvefile(planet_name, mission="TESS", author="SPOC", cadence="short").download_all()
        # lc_files = search_lightcurvefile(target_name, mission="TESS").download_all()
        # if lc_files is None or len(lc_files) == 0:
        #     raise SystemExit("No Kepler lightcurve files found for: " + planet_name)

        if lc_files is None or len(lc_files) == 0:
            print(f"No Kepler lightcurve files found for: {planet_name}, skipping.")
            return None
        else:
            print(f"Found {len(lc_files)} files. Stitching ...")

        combined = lc_files.stitch()
        # lcs = [f.PDCSAP_FLUX for f in lc_files if f.PDCSAP_FLUX is not None]
        # combined = lcs[0]
        # if len(lcs) > 1:
        #     for lc in lcs[1:]:
        #         combined = combined.append(lc)

        # --- CLEAN DATA ---
        lc_clean = combined.remove_nans().remove_outliers(sigma=5)
        time = lc_clean.time.value  # BTJD (BJD - 2457000)
        flux = lc_clean.flux.value
        flux_err = lc_clean.flux_err.value

        df_full = pd.DataFrame({"time_btjd": time, "flux": flux, "flux_err": flux_err})
        # df_full.to_csv("WASP-18_TESS_full_PDCSAP.csv", index=False)
        print("Fully cleaned light curve.")

        # --- EXTRACT ONE TRANSIT WINDOW (no interpolation) ---
        # max_missing = 0.1   # allow up to 10% missing data
        epoch = 0
        n_windows = 2
        df_transit = None

        while df_transit is None:
            # find mid-transit time for this epoch
            t0_epoch = t0_btjd + epoch * period_days

            # define window: center ± (duration + 1 duration each side)
            half_window = n_windows * window
            mask = (time > t0_epoch - half_window) & (time < t0_epoch + half_window)

            sel_time = time[mask]
            sel_flux = flux[mask]
            sel_err = flux_err[mask]

            # if more than 250 points, just take central 250 around transit
            if len(sel_time) > 250:
                mid_idx = np.argmin(np.abs(sel_time - t0_epoch))
                half_n = 125
                start = max(0, mid_idx - half_n)
                end = start + 250
                sel_time = sel_time[start:end]
                sel_flux = sel_flux[start:end]
                sel_err = sel_err[start:end]

                df_transit = pd.DataFrame({"time_btjd": sel_time, "flux": sel_flux, "flux_err": sel_err})
            else:
                # not enough data, try next transit
                epoch += 1
                n_windows += 1
                if epoch > 1000:  # safety break
                    raise RuntimeError("No transit found with enough data")

        print(f"Returning one transit with {len(sel_time)} raw points.")
        return df_transit

        # # --- EXTRACT TRANSIT WINDOWS ---
        # # Convert T0 into BTJD (TESS time system)
        # # t0_btjd = t0_bjd - 2457000.0
        # phases = ((time - t0_btjd) / period_days) % 1.0
        # in_transit_mask = (phases < window/period_days) | (phases > 1 - window/period_days)

        # df_transits = pd.DataFrame({
        #     "time_btjd": time[in_transit_mask],
        #     "flux": flux[in_transit_mask],
        #     "flux_err": flux_err[in_transit_mask],
        #     "phase": phases[in_transit_mask]
        # })
        # # df_transits.to_csv("WASP-18_TESS_transit_windows.csv", index=False)
        # print("Returning transit windows.")
        # return df_transits

    def find_data_tess(self, planet_name, period_days, t0_btjd, window, points=250, cadence="short"):
        tid = planet_name
        print(f"Searching TESS lightcurves for {planet_name} (TIC {tid}) ...")

        if cadence == "short":
            print("Searching for 2-min cadence...")
            lc_files = lk.search_lightcurve(f"TIC {tid}", author="SPOC", cadence="short").download_all()
        else:
            print("Searching for any cadence...")
            lc_files = lk.search_lightcurve(f"TIC {tid}", author="SPOC").download_all()

        if lc_files is None or len(lc_files) == 0:
            print(f"No TESS lightcurve files found for: {planet_name}, skipping.")
            return None

        print(f"Found {len(lc_files)} files. Stitching ...")
        combined = lc_files.stitch()

        # --- CLEAN DATA ---
        lc_clean = combined.remove_nans().remove_outliers(sigma=5)
        time = lc_clean.time.value
        flux = lc_clean.flux.value
        flux_err = lc_clean.flux_err.value

        # --- EXTRACT ONE TRANSIT WINDOW ---
        epoch = 0
        n_windows = 2
        df_transit = None
        while df_transit is None:
            t0_epoch = t0_btjd + epoch * period_days
            half_window = n_windows * window
            mask = (time > t0_epoch - half_window) & (time < t0_epoch + half_window)

            sel_time = time[mask]
            sel_flux = flux[mask]
            sel_err = flux_err[mask]

            if len(sel_time) > points:
                mid_idx = np.argmin(np.abs(sel_time - t0_epoch))
                start = max(0, mid_idx - int(points / 2))
                end = start + points
                sel_time = sel_time[start:end]
                sel_flux = sel_flux[start:end]
                sel_err = sel_err[start:end]

                df_transit = pd.DataFrame({"time_btjd": sel_time, "flux": sel_flux, "flux_err": sel_err})
            else:
                epoch += 1
                n_windows = 3
                if epoch > 1000:
                    raise RuntimeError("No transit found with enough data")

        print(f"Returning one TESS transit for {planet_name} with {len(df_transit)} points.")
        return df_transit
