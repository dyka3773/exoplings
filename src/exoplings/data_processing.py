import pandas as pd


def load_data(filepath) -> pd.DataFrame:
    """Load data from various file formats."""
    # TODO: Add support for planet through John's module
    try:
        file_ext = filepath.rsplit(".", 1)[1].lower()
        if file_ext == "csv":
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")
