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
    try:
        if set(["object_id", "observation_date", "brightness"]).issubset(df.columns):
            fig = px.scatter(
                df,
                x="observation_date",
                y="brightness",
                error_y="brightness_error" if "brightness_error" in df.columns else None,
                color="object_id",
                title="Light Curve",
                labels={
                    "observation_date": "Observation Date",
                    "brightness": "Brightness",
                },
            )
            fig.update_traces(mode="lines+markers")
            fig.update_layout(height=600, showlegend=True, hovermode="closest")
            return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating plot: {str(e)}", x=0.5, y=0.5, showarrow=False)
        return fig
