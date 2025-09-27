import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_interactive_plot(df: pd.DataFrame) -> go.Figure:
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
                "flux": "Flux",
            },
        )
        fig.update_traces(mode="lines+markers")
        fig.update_layout(height=600, showlegend=True, hovermode="closest")
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating plot: {str(e)}", x=0.5, y=0.5, showarrow=False)
        return fig


def create_posterior_plot(df: pd.DataFrame) -> go.Figure:
    """Create the posterior plot for exoplanet prediction.

    Args:
        df (pd.DataFrame): DataFrame containing light curve data.

    Returns:
        go.Figure: Plotly figure object for posterior plot.
    """
    # TODO: Implement actual posterior plot creation logic
    return go.Figure()
