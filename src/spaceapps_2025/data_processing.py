import pandas as pd
import random



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
    
    
def predict_is_exoplanet(light_curve_df):
    is_exoplanet = random.choice([True, False])
    certainty = round(random.uniform(50, 99.9), 2)
    
    exoplanet_result = {
        "is_exoplanet": is_exoplanet,
        "certainty": certainty,
    }
    return exoplanet_result