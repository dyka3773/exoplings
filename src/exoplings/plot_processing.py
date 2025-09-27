import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import swyft
import torch
from plotly.graph_objs._figure import Figure
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline
from swyft.plot.plot import _get_HDI_thresholds, get_pdf

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


def plot_corner_plotly(lrs_coll, parnames, labels=None, truth=None, bins=100, smooth=0.0, figsize=(600, 600)):
    """
    Minimal Plotly corner plot for swyft inference results.
    """
    K = len(parnames)
    fig: Figure = make_subplots(rows=K, cols=K, shared_xaxes=False, shared_yaxes=False, horizontal_spacing=0.02, vertical_spacing=0.02)

    if labels is None:
        labels = parnames

    for i in range(K):
        for j in range(K):
            if i < j:
                continue  # upper triangle blank

            # 1D marginal (diagonal)
            if i == j:
                v, zm = get_pdf(lrs_coll, parnames[i], bins=bins, smooth=smooth)
                zm = zm[:, 0]

                # Density curve
                fig.add_trace(go.Scatter(x=zm, y=v, mode="lines", line=dict(color="black")), row=i + 1, col=j + 1)

                # Credible interval shading (HDI bands)
                levels = sorted(_get_HDI_thresholds(v, cred_level=[0.68268, 0.95450, 0.99730]))
                y0, y1 = -0.05 * v.max(), 1.1 * v.max()
                shades = ["whitesmoke", "gainsboro", "silver"]

                for k, lvl in enumerate(levels):
                    mask = v >= lvl
                    if mask.any():
                        lower, upper = zm[mask].min(), zm[mask].max()
                        fig.add_shape(
                            type="rect",
                            x0=lower,
                            x1=upper,
                            y0=y0,
                            y1=y1,
                            fillcolor=shades[k],
                            line=dict(color="rgba(0,0,0,0)"),
                            opacity=0.3,
                            layer="below",
                            row=i + 1,
                            col=j + 1,
                        )

                # Density curve
                fig.add_trace(go.Scatter(x=zm, y=v, mode="lines", line=dict(color="black")), row=i + 1, col=j + 1)

                # True value as red dashed line
                if truth and parnames[i] in truth:
                    fig.add_shape(
                        type="line",
                        x0=truth[parnames[i]],
                        x1=truth[parnames[i]],
                        y0=y0,
                        y1=y1,
                        line=dict(color="red", dash="dash"),
                        row=i + 1,
                        col=j + 1,
                    )

                # True value as red dashed line
                if truth and parnames[i] in truth:
                    fig.add_trace(
                        go.Scatter(
                            x=[truth[parnames[i]], truth[parnames[i]]],
                            y=[y0, y1],
                            mode="lines",
                            line=dict(color="red", dash="dash"),
                            showlegend=False,
                        ),
                        row=i + 1,
                        col=j + 1,
                    )

            # 2D joint posterior (lower triangle)
            if j < i:
                counts, xy = get_pdf(lrs_coll, [parnames[j], parnames[i]], bins=bins, smooth=smooth)
                xbins = xy[:, 0]
                ybins = xy[:, 1]

                # Heatmap for smooth density
                fig.add_trace(go.Heatmap(z=counts.T, x=xbins, y=ybins, colorscale="Greys", showscale=False), row=i + 1, col=j + 1)

                # Contour lines for HDI levels
                levels = sorted(_get_HDI_thresholds(counts, cred_level=[0.68268, 0.95450, 0.99730]))
                fig.add_trace(
                    go.Contour(
                        z=counts.T,
                        x=xbins,
                        y=ybins,
                        contours=dict(
                            start=levels[0],
                            end=levels[-1],
                            size=(levels[-1] - levels[0]) / len(levels),
                            coloring="none",  # just lines, no fill
                        ),
                        line=dict(color="black", width=1),
                        showscale=False,
                    ),
                    row=i + 1,
                    col=j + 1,
                )

                if truth:
                    if parnames[j] in truth and parnames[i] in truth:
                        fig.add_trace(
                            go.Scatter(
                                x=[truth[parnames[j]]],
                                y=[truth[parnames[i]]],
                                mode="markers",
                                marker=dict(color="red", size=8, symbol="x"),
                                name="truth",
                            ),
                            row=i + 1,
                            col=j + 1,
                        )

            # Axis labels
            if i == K - 1:
                fig.update_xaxes(title_text=labels[j], row=i + 1, col=j + 1)
            if j == 0 and i > 0:
                fig.update_yaxes(title_text=labels[i], row=i + 1, col=j + 1)

    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        template="simple_white",
        showlegend=False,
        font=dict(size=18),  # increase font size here
    )

    fig.update_layout(margin=dict(l=50, r=50, t=50, b=50))

    return fig


def plot_smart_multiD_infer(network, trainer) -> Figure:
    z_true = [0.1, 1.0, 88.0, 0.0]

    # Run simulation
    mysampless = simulator.sample(conditions={"z": z_true})
    null_xs = mysampless["x"]

    prior_samples = simulator.sample(targets=["x"], N=10000)
    prior_samples["z"] = prior_samples["z"].astype(np.float32)

    predictions = trainer.infer(network, swyft.Sample(x=null_xs), prior_samples)

    # Build Plotly corner plot
    fig_post = plot_corner_plotly(
        predictions,
        ["z[0]", "z[1]", "z[2]", "z[3]"],
        labels=[r"$r_p/r_*$", "T [arbitrary units]", r"$i$ [deg]", r"$t_0$ [arbitrary units]"],
        truth={"z[0]": z_true[0], "z[1]": z_true[1], "z[2]": z_true[2], "z[3]": z_true[3]},
        bins=200,
        smooth=3,
        figsize=(1800, 1800),
    )
    return fig_post
