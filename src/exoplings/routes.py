import json
import os
import time

import plotly.utils
from flask import flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from .data_processing import load_data, predict_is_exoplanet
from .plot_processing import create_interactive_plot
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
            df = load_data(filepath)
            data_info = {
                "filename": filename,
            }
            results = None

            light_curve_fig = create_interactive_plot(df)  # FIXME: this is a placeholder function, to be changed when we have real data

            posterior_fig, results = predict_is_exoplanet(df)

            light_curve_plot_json = json.dumps(light_curve_fig, cls=plotly.utils.PlotlyJSONEncoder)
            posterior_plot_json = json.dumps(posterior_fig, cls=plotly.utils.PlotlyJSONEncoder)

            return render_template(
                "visualize.html",
                light_curve_plot_json=light_curve_plot_json,
                posterior_plot_json=posterior_plot_json,
                data_info=data_info,
                exoplanet_result=results,
                most_recent_curves=get_most_recent_curves(app.config["UPLOAD_FOLDER"], limit=10),
            )
        except Exception as e:
            flash(f"Error visualizing data: {str(e)}")
            return redirect(url_for("index"))
