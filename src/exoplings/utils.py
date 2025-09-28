import pathlib

import numpy as np
import torch


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "csv",
    }


def get_most_recent_curves(directory=".uploads", limit=10):
    """Retrieve the most recent light curve files from the upload directory.

    Args:
        directory (str): The directory to search for uploaded files.
        limit (int): The maximum number of recent files to return.
    Returns:
        List[str]: List of the most recent file names.
    """
    p = pathlib.Path(directory)
    files = sorted(p.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
    return [f.name for f in files[:limit] if f.is_file()]


def compute_credible_intervals(z_values, density, levels=[0.682, 0.954, 0.997]):
    """Compute highest density intervals (HDIs) for given credible levels."""
    # Flatten and sort by density descending
    density_flat = density.flatten()
    z_flat = z_values.flatten()
    # Convert to NumPy and sort descending safely
    # sorted_indices = np.argsort(density_flat.numpy())  # sort ascending first
    # sorted_indices = sorted_indices[::-1].copy()       # reverse with a copy
    sorted_indices = torch.argsort(density_flat, descending=True)

    density_sorted = density_flat[sorted_indices]

    total_density = density_sorted.sum()
    cumulative_density = np.cumsum(density_sorted)

    intervals = []
    for level in levels:
        # Find threshold density that includes the desired probability mass
        idx = np.argmax(cumulative_density >= total_density * level)
        threshold = density_sorted[idx]

        # Include all z-values with density >= threshold
        mask = density_flat >= threshold
        interval_min = z_flat[mask].min()
        interval_max = z_flat[mask].max()
        intervals.append((interval_min, interval_max))

    return intervals


def compute_cdf(density):
    # Compute cumulative density function (normalized)
    cdf = np.cumsum(density) / torch.sum(density)

    return cdf
