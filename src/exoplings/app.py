import os

import torch
from flask import Flask
from swyft import SwyftTrainer

from .models.network import Network
from .models.simulator import Simulator

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


# globals needed
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
network = Network()
network.load_state_dict(torch.load(os.path.join(current_dir, "ai_models", "CNN_1D.pth"), weights_only=True))

simulator = Simulator(rand_inc=True, rand_t0=True, rand_per=True, t_len=250)
trainer = SwyftTrainer(accelerator=DEVICE)

# Register routes from routes.py
from .routes import register_routes

register_routes(app)


def main():
    """Main entry point for the application."""
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(debug=debug, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
