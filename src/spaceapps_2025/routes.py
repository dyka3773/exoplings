import json
import os
import random

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
from flask import flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "csv",
        "xlsx",
        "xls",
        "json",
    }


def load_data(filepath):
    """Load data from various file formats."""
    try:
        file_ext = filepath.rsplit(".", 1)[1].lower()
        if file_ext == "csv":
            df = pd.read_csv(filepath)
        elif file_ext in ["xlsx", "xls"]:
            df = pd.read_excel(filepath)
        elif file_ext == "json":
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def detect_exoplanet(light_curve_df):
    is_exoplanet = random.choice([True, False])
    certainty = round(random.uniform(50, 99.9), 2)
    return is_exoplanet, certainty


def create_interactive_plot(df):
    try:
        if set(["object_id", "observation_date", "brightness"]).issubset(df.columns):
            fig = px.scatter(
                df,
                x="observation_date",
                y="brightness",
                error_y="brightness_error"
                if "brightness_error" in df.columns
                else None,
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
        fig.add_annotation(
            text=f"Error creating plot: {str(e)}", x=0.5, y=0.5, showarrow=False
        )
        return fig


def register_routes(app):
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.route("/upload", methods=["POST"])
    def upload_file():
        if (
            "file" not in request.files
            or request.files["file"] is None
            or request.files["file"] == ""
        ):
            flash("No file selected")
            return redirect(request.url)
        file = request.files["file"]
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            try:
                df = load_data(filepath)
                flash(
                    f"File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns."
                )
                return redirect(url_for("visualize", filename=filename))
            except Exception as e:
                flash(f"Error processing file: {str(e)}")
                os.remove(filepath)
                return redirect(request.url)
        else:
            flash("Invalid file type. Please upload a CSV, Excel, or JSON file.")
            return redirect(request.url)

    @app.route("/visualize/<filename>")
    def visualize(filename):
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(filepath):
            flash("File not found")
            return redirect(url_for("index"))
        try:
            df = load_data(filepath)
            data_info = {
                "filename": filename,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
            }
            exoplanet_result = None
            if set(["object_id", "observation_date", "brightness"]).issubset(
                df.columns
            ):
                is_exoplanet, certainty = detect_exoplanet(df)
                exoplanet_result = {
                    "is_exoplanet": is_exoplanet,
                    "certainty": certainty,
                }
            fig = create_interactive_plot(df)
            plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return render_template(
                "visualize.html",
                plot_json=plot_json,
                data_info=data_info,
                exoplanet_result=exoplanet_result,
            )
        except Exception as e:
            flash(f"Error visualizing data: {str(e)}")
            return redirect(url_for("index"))
            return redirect(url_for("index"))
