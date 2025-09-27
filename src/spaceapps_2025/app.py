import os

from flask import Flask

from .routes import register_routes

current_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(current_dir, "templates"),
    static_folder=os.path.join(current_dir, "static"),
)
app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key-change-this-in-production")

# Configuration
UPLOAD_FOLDER = ".uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Register routes from routes.py
register_routes(app)


def main():
    """Main entry point for the application."""
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(debug=debug, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
