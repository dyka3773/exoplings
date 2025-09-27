import json
import os
import time

import plotly.utils
import swyft
import torch
from flask import flash, redirect, render_template, request, url_for
from plotly.graph_objs._figure import Figure
from werkzeug.utils import secure_filename

from .app import (
    multi_d_network as network_multi,
    one_d_network as network,
    trainer,
)
from .data_processing import load_data
from .PlanetDetailExtractor import PlanetDetailExtractor
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

    @app.route("/upload", methods=["POST"])
    def upload_file():
        """Upload and process a light curve data file.

        Returns:
            Redirect to visualization page or back to upload with error message.
        """
        if "file" not in request.files or request.files["file"] is None or request.files["file"] == "":
            flash("No file selected")
            return redirect(request.url)

        file = request.files["file"]

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # add timestamp to filename to avoid overwriting
            filename = f"{int(time.time())}_{filename}"

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                df = load_data(filepath)
                flash(f"File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
                return redirect(url_for("visualize", filename=filename))

            except Exception as e:
                flash(f"Error processing file: {str(e)}")
                os.remove(filepath)
                return redirect(request.url)
        else:
            flash("Invalid file type. Please upload a CSV file.")
            return redirect(request.url)

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
            flash("File not found")
            return redirect(url_for("index"))
        try:
            # df = load_data(filepath)

            # planet_name = myplanets.confirmed_planets().sample(1)["tid"].values[0]  # tid, toi
            planet_name = 432549364

            planet_extractor = PlanetDetailExtractor(telescope="tess")
            planet_params = planet_extractor.find_planet_details(planet_name)

            tempdf = planet_extractor.find_data_tess(
                planet_name,
                period_days=planet_params["per"],
                t0_btjd=planet_params["t0"],
                window=planet_params["duration"],
                points=250,
                cadence="short",
            )
            light_curve_fig: Figure = create_simple_lc_plot(tempdf)

            real_test = tempdf["flux"].values

            data_info = {
                "filename": planet_name,
            }

            prior_samples = swyft.Samples({"z": torch.linspace(0.0, 0.3, 10000)})

            starting_time = time.perf_counter()
            predictions = trainer.infer(network, swyft.Sample(x=real_test), prior_samples)
            end_time = time.perf_counter()

            processing_time = int((end_time - starting_time) * 1000)  # in milliseconds

            conversion_factor = 1 / (0.022 * 24.0) * 3.2
            z_true = [planet_params["z"], planet_params["duration"] * conversion_factor, 90.0, 0.0]

            posterior_fig, credible_intervals, mode, certainty, is_exoplanet = create_posterior_1D_plot(z_true, predictions)
            posterior_lc_fig = create_posterior_lc_plot(z_true, real_test, credible_intervals, mode)

            # TODO: re-enable when model is ready
            posterior_corner_fig = create_posterior_lc_plot(z_true, real_test, credible_intervals, mode)
            # posterior_corner_fig = plot_smart_multiD_infer(network_multi, trainer)

            light_curve_plot_json = json.dumps(light_curve_fig, cls=plotly.utils.PlotlyJSONEncoder)
            posterior_plot_json = json.dumps(posterior_fig, cls=plotly.utils.PlotlyJSONEncoder)
            posterior_lc_plot_json = json.dumps(posterior_lc_fig, cls=plotly.utils.PlotlyJSONEncoder)
            posterior_corner_plot_json = json.dumps(posterior_corner_fig, cls=plotly.utils.PlotlyJSONEncoder)

            return render_template(
                "visualize.html",
                light_curve_plot_json=light_curve_plot_json,
                posterior_plot_json=posterior_plot_json,
                posterior_lc_plot_json=posterior_lc_plot_json,
                corner_plot_json=posterior_corner_plot_json,
                data_info=data_info,
                exoplanet_result={"is_exoplanet": is_exoplanet, "certainty": certainty},
                most_recent_curves=get_most_recent_curves(app.config["UPLOAD_FOLDER"], limit=10),
                processing_time=processing_time,
            )
        except Exception as e:
            flash(f"Error visualizing data: {str(e)}")
            return redirect(url_for("index"))
