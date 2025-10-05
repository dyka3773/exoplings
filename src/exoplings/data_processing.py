import pathlib

import pandas as pd

from .app import app
from .PlanetDetailExtractor import PlanetDetailExtractor

tess_planet_extractor = PlanetDetailExtractor(telescope="tess")
kepler_planet_extractor = PlanetDetailExtractor(telescope="kepler")


def load_data(data) -> tuple[pd.DataFrame, dict]:
    """Load data from a file path or identifier.

    Args:
        data (str | int): File path to CSV file or integer ID for TESS/Kepler data.

    Returns:
        tuple[pd.DataFrame, dict]: DataFrame with light curve data and dictionary with planet parameters.
    """
    possible_uploaded_path = pathlib.Path(app.config["UPLOAD_FOLDER"]) / str(data)
    if possible_uploaded_path.exists() and possible_uploaded_path.is_file() and possible_uploaded_path.suffix.lower() == ".csv":
        df = pd.read_csv(possible_uploaded_path)
        return df, {
            "z": None,
            "duration": None,
            "impact": None,
        }

    else:
        try:
            planet_id = int(data)
            planet_params = tess_planet_extractor.find_planet_details(planet_id)

            df = tess_planet_extractor.find_data_tess(
                planet_id,
                period_days=planet_params["per"],
                t0_btjd=planet_params["t0"],
                window=planet_params["duration"],
            )

            if df is None or df.empty:
                raise ValueError(f"No TESS data found for identifier: {data}")

            return df, planet_params
        except (ValueError, TypeError):
            planet_id = str(data)
            planet_params = kepler_planet_extractor.find_planet_details(planet_id)

            if planet_params is None:
                raise ValueError(f"No planet details found for identifier: {data}")

            df = kepler_planet_extractor.find_data_kepler(
                planet_id,
                period_days=planet_params["per"],
                t0_btjd=planet_params["t0"],
                window=planet_params["duration"],
            )

            if df is None or df.empty:
                raise ValueError(f"No Kepler data found for identifier: {data}")

            return df, planet_params
