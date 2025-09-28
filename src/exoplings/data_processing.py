import pathlib

import pandas as pd

from .PlanetDetailExtractor import PlanetDetailExtractor

planet_extractor = PlanetDetailExtractor(telescope="tess")


def load_data(filepath) -> tuple[pd.DataFrame, dict]:
    """Load data from various file formats."""
    if isinstance(filepath, int):
        planet_params = planet_extractor.find_planet_details(filepath)

        print(planet_params)

        df = planet_extractor.find_data_tess(
            filepath,
            period_days=planet_params["per"],
            t0_btjd=planet_params["t0"],
            window=planet_params["duration"],
            points=250,
            cadence="short",
        )

        return df, planet_params

    elif isinstance(filepath, str) and pathlib.Path(filepath).is_file():
        file_ext = filepath.rsplit(".", 1)[1].lower()
        if file_ext == "csv":
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        return df, {
            "z": None,
            "duration": None,
        }

    return pd.DataFrame(), dict()  # Return empty DataFrame if no valid data is loaded
