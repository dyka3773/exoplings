import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.graph_objs._figure import Figure
from scipy.interpolate import CubicSpline

from .app import simulator
from .utils import compute_cdf, compute_credible_intervals


def create_simple_lc_plot(df: pd.DataFrame) -> go.Figure:
    """Create an interactive plotly figure from the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing light curve data.

    Returns:
        go.Figure: Plotly figure object.
    """
    # Support WASP-18_TESS format: columns 'time_btjd', 'flux'
    try:
        fig: go.Figure = px.scatter(
            df,
            x="time_btjd",
            y="flux",
            title="Light Curve",
            labels={
                "time_btjd": "Time (BTJD)",
                "flux": "Normalized Flux",
            },
        )
        fig.update_traces(mode="lines+markers")
        fig.update_layout(height=600, showlegend=True, hovermode="closest")
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating plot: {str(e)}", x=0.5, y=0.5, showarrow=False)
        return fig


def create_posterior_1D_plot(z_true, predictions) -> tuple[Figure, list[tuple[float, float]], float, float, bool]:
    # Extract posterior samples and density (and sort)
    z_values = predictions.params.T[0][0]
    z_values, indices = torch.sort(z_values)
    density = np.exp(predictions.logratios.T[0])[indices]

    credible_intervals: list[tuple[float, float]] = compute_credible_intervals(z_values, density)

    # Build figure
    fig_post = go.Figure()

    # Plot density curve
    fig_post.add_trace(go.Scatter(x=z_values, y=density, mode="lines", line=dict(color="black"), name="Density"))

    # Get y-limits
    y_min, y_max = 0, max(density) * 1.1

    # Shade credible intervals as vertical rectangles
    shades = ["black", "grey", "whitesmoke"]  # High contrast greys
    for j, (lower, upper) in enumerate(credible_intervals):
        fig_post.add_shape(
            type="rect", x0=lower, x1=upper, y0=y_min, y1=y_max, fillcolor=shades[j], line=dict(color="dimgrey"), opacity=0.3, layer="below"
        )

    # Compute intervals
    tensor_credint = torch.tensor(credible_intervals)
    dlow = tensor_credint[0, 0] - tensor_credint[2, 0]
    dhigh = -tensor_credint[0, 1] + tensor_credint[2, 1]
    zmax = z_values[np.argmax(density)]

    # Layout adjustments
    fig_post.update_layout(
        xaxis=dict(title="rₚ/r<sub>s</sub>", range=[0, float(zmax + 3 * dhigh)]),
        yaxis=dict(title="Probability density", range=[y_min, y_max]),
        template="simple_white",
    )

    # Add vertical line at true z
    fig_post.add_vline(x=z_true[0], line=dict(color="red"))

    # Add x-axis label
    fig_post.update_xaxes(title="rₚ/r<sub>s</sub>")

    fig_post.add_trace(go.Scatter(x=[z_true[0], z_true[0]], y=[y_min, y_max], mode="lines", line=dict(color="red", dash="dash"), name="True value"))
    fig_post.update_xaxes(range=[0, min(zmax + 3 * dhigh, 0.3)])

    mode: float = z_values[torch.argmax(density)].item()

    # compute certainty and is_exoplanet
    cdf = compute_cdf(density)
    cs = CubicSpline(z_values, cdf)
    z_cutoff = 0.05  # test and change that
    certainty = cs(z_cutoff)
    if certainty >= 0.9:
        is_exoplanet = False
    else:
        is_exoplanet = True
        certainty = 1 - certainty

    return fig_post, credible_intervals, mode, certainty, is_exoplanet


def create_posterior_lc_plot(z_true, null_xs, credible_intervals, mode):
    # Compute min/max light curves
    min_zpred, max_zpred = credible_intervals[0]
    min_lc = simulator.sample(conditions={"z": [min_zpred, z_true[1], z_true[2], z_true[3]]})["m"]
    max_lc = simulator.sample(conditions={"z": [max_zpred, z_true[1], z_true[2], z_true[3]]})["m"]
    mode_lc = simulator.sample(conditions={"z": [mode, z_true[1], z_true[2], z_true[3]]})["m"]

    # X-axis
    x_vals = np.arange(len(null_xs))

    # Build figure
    fig_lc = go.Figure()

    # Shaded credible region (min_lc to max_lc)
    fig_lc.add_trace(
        go.Scatter(
            x=x_vals,
            y=max_lc,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),  # invisible line
            showlegend=False,
        )
    )
    fig_lc.add_trace(
        go.Scatter(
            x=x_vals,
            y=min_lc,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),  # invisible line
            fill="tonexty",  # fill area between min_lc and max_lc
            fillcolor="gainsboro",
            opacity=1.0,
            name="Credible region",
        )
    )

    # Null samples (baseline)
    fig_lc.add_trace(go.Scatter(x=x_vals, y=null_xs, mode="lines", line=dict(color="black"), opacity=0.5, name="Null samples"))

    # Observed / sampled light curve
    fig_lc.add_trace(go.Scatter(x=x_vals, y=mode_lc, mode="lines", line=dict(color="black"), name="Mode LC"))

    # Layout
    fig_lc.update_layout(xaxis_title="Arbitrary Time", yaxis_title="Normalized Flux", template="simple_white")

    return fig_lc
