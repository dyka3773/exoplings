import json
import os
import random

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

# Get the directory where this file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(current_dir, "templates"),
    static_folder=os.path.join(current_dir, "static"),
)
app.secret_key = "your-secret-key-change-this-in-production"

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "json"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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


# Placeholder function for exoplanet detection
def detect_exoplanet(light_curve_df):
    """
    Randomly determines if the light curve is a possible exoplanet and returns a certainty percentage.
    """
    is_exoplanet = random.choice([True, False])
    certainty = round(random.uniform(50, 99.9), 2)  # Certainty between 50% and 99.9%
    return is_exoplanet, certainty


def create_interactive_plot(df):
    """Create an interactive plot based on the uploaded data."""
    try:
        # If the file looks like a light curve, plot it as such
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
        # Return an error plot
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating plot: {str(e)}", x=0.5, y=0.5, showarrow=False
        )
        return fig


@app.route("/")
def index():
    """Home page with file upload form."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and redirect to visualization page."""
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
            # Test if the file can be loaded
            df = load_data(filepath)
            flash(
                f"File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns."
            )
            return redirect(url_for("visualize", filename=filename))
        except Exception as e:
            flash(f"Error processing file: {str(e)}")
            os.remove(filepath)  # Clean up the file
            return redirect(request.url)
    else:
        flash("Invalid file type. Please upload a CSV, Excel, or JSON file.")
        return redirect(request.url)


@app.route("/visualize/<filename>")
def visualize(filename):
    """Display the interactive visualization page."""
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if not os.path.exists(filepath):
        flash("File not found")
        return redirect(url_for("index"))

    try:
        df = load_data(filepath)

        # Create data summary
        data_info = {
            "filename": filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
        }

        # If it's a light curve, run exoplanet detection
        exoplanet_result = None
        if set(["object_id", "observation_date", "brightness"]).issubset(df.columns):
            is_exoplanet, certainty = detect_exoplanet(df)
            exoplanet_result = {
                "is_exoplanet": is_exoplanet,
                "certainty": certainty,
            }

        # When generating plots using swyft, see also this:
        # https://stackoverflow.com/a/54947575/15552149

        # Create the plot
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


def main():
    """Main entry point for the application."""
    app.run(debug=True, host="127.0.0.1", port=5000)


if __name__ == "__main__":
    main()
