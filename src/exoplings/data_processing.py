import random

import pandas as pd
import plotly.graph_objects as go

from .plot_processing import create_posterior_plot


def load_data(filepath) -> pd.DataFrame:
    """Load data from various file formats."""
    try:
        file_ext = filepath.rsplit(".", 1)[1].lower()
        if file_ext == "csv":
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def predict_is_exoplanet(light_curve_df) -> tuple[go.Figure, dict]:
    """Predict if the light curve indicates an exoplanet.

    Args:
        light_curve_df (pd.DataFrame): DataFrame containing light curve data.

    Returns:
        Tuple[go.Figure, dict]: A placeholder figure and a dictionary with prediction results.
    """
    is_exoplanet = random.choice([True, False])
    certainty = round(random.uniform(50, 99.9), 2)

    exoplanet_result = {
        "is_exoplanet": is_exoplanet,
        "certainty": certainty,
    }

    fig = create_posterior_plot(light_curve_df)

    # TODO: Delete this when the above is ready
    fig.add_annotation(
        text=f"Prediction: {'Exoplanet' if is_exoplanet else 'No Exoplanet'}<br>Certainty: {certainty}%",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20),
    )

    return fig, exoplanet_result
