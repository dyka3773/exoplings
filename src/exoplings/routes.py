import json
import os
import time
from pathlib import Path

import plotly.utils
import swyft
import torch
from flask import flash, redirect, render_template, request, url_for
from plotly.graph_objs._figure import Figure
from werkzeug.utils import secure_filename

from .app import multi_d_network as network_multi
from .app import one_d_network as network
from .app import trainer
from .data_processing import load_data
from .plot_processing import create_posterior_1D_plot, create_posterior_lc_plot, create_simple_lc_plot, plot_smart_multiD_infer
from .utils import allowed_file, get_most_recent_curves


def register_routes(app):
    @app.route("/")
    def index():
        """Render the home page.

        Returns:
            Rendered index.html template.
        """
        return render_template(
            "index.html",
            most_recent_curves=get_most_recent_curves(app.config["UPLOAD_FOLDER"], limit=10),
        )

    @app.route("/about")
    def about():
        """Render the about page.

        Returns:
            Rendered about.html template.
        """
        return render_template("about.html")

    @app.route("/model")
    def model():
        """Render the model page.

        Returns:
            Rendered model.html template.
        """
        return render_template("model.html")

    @app.route("/upload", methods=["POST"])
    def upload_file():
        """Upload and process a light curve data file.

        Returns:
            Redirect to visualization page or back to upload with error message.
        """
        if "file" not in request.files or request.files["file"] is None or request.files["file"] == "":
            flash("No file selected")
            return redirect(url_for("index"))

        file = request.files["file"]

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # add timestamp to filename to avoid overwriting
            filename = f"{int(time.time())}_{filename}"

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                df, _ = load_data(filepath)
                flash(f"File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
                return redirect(url_for("visualize", filename=filename))

            except Exception as e:
                flash(f"Error processing file: {str(e)}")
                os.remove(filepath)
                return redirect(url_for("index"))
        else:
            flash("Invalid file type. Please upload a CSV file.")
            return redirect(url_for("index"))

    @app.route("/visualize/<filename>")
    def visualize(filename):
        """Visualize the uploaded light curve data.

        Args:
            filename (str): The name of the uploaded file.

        Returns:
            Rendered visualize.html template with plot and data info.
        """

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        if not os.path.exists(filepath):
            try:
                # In case filename is an integer, we assume it's a planet ID, not a file, so we convert it
                # If this fails, we flash an error and redirect
                filepath = int(filename)
            except ValueError:
                flash("File not found")
                return redirect(url_for("index"))

        try:
            df, planet_params = load_data(filepath)

            light_curve_fig: Figure = create_simple_lc_plot(df)

            real_test = df["flux"].values.astype("float32")

            data_info = {
                "filename": Path(filepath).name if isinstance(filepath, str) else f"Planet ID: {filepath}",
            }

            prior_samples = swyft.Samples({"z": torch.linspace(0.0, 0.3, 10000)})

            starting_time = time.perf_counter()
            predictions = trainer.infer(network, swyft.Sample(x=real_test), prior_samples)
            end_time = time.perf_counter()

            processing_time = int((end_time - starting_time) * 1000)  # in milliseconds

            conversion_factor = 1 / (0.022 * 24.0) * 3.2
            z_true = [
                planet_params["z"],
                planet_params["duration"] * conversion_factor if planet_params["duration"] else 100,
                90.0,
                0.0,
            ]

            posterior_fig, credible_intervals, mode, certainty, is_exoplanet = create_posterior_1D_plot(z_true, predictions)
            posterior_lc_fig = None
            # in case of CSV do not produce posterior lc plot because of missing true values
            if planet_params["z"]:
                posterior_lc_fig: Figure = create_posterior_lc_plot(z_true, real_test, credible_intervals, mode)

            posterior_corner_fig = plot_smart_multiD_infer(network_multi, trainer)

            light_curve_plot_json = json.dumps(light_curve_fig, cls=plotly.utils.PlotlyJSONEncoder)
            posterior_plot_json = json.dumps(posterior_fig, cls=plotly.utils.PlotlyJSONEncoder)
            posterior_lc_plot_json = json.dumps(posterior_lc_fig, cls=plotly.utils.PlotlyJSONEncoder)
            posterior_corner_plot_json = json.dumps(posterior_corner_fig, cls=plotly.utils.PlotlyJSONEncoder)

            return render_template(
                "visualize.html",
                light_curve_plot_json=light_curve_plot_json,
                posterior_plot_json=posterior_plot_json,
                posterior_lc_plot_json=posterior_lc_plot_json if posterior_lc_fig else None,
                corner_plot_json=posterior_corner_plot_json,
                data_info=data_info,
                exoplanet_result={"is_exoplanet": is_exoplanet, "certainty": certainty},
                most_recent_curves=get_most_recent_curves(app.config["UPLOAD_FOLDER"], limit=10),
                processing_time=processing_time,
            )
        except Exception as e:
            flash(f"Error visualizing data: {str(e)}")
            return redirect(url_for("index"))
            flash(f"Error visualizing data: {str(e)}")
            return redirect(url_for("index"))
