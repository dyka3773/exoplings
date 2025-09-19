import pathlib


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "csv",
        "xlsx",
        "xls",
        "json",
    }


def get_most_recent_curves(directory="uploads", limit=10):
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
