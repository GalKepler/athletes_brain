import pandas as pd
from pathlib import Path
import pickle
import numpy as np


def save_results(df: pd.DataFrame, output_dir: Path, filename: str):
    """
    Saves a DataFrame of results to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    output_dir : Path
        The directory where the file will be saved.
    filename : str
        The name of the CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / filename, index=False)
    print(f"Results saved to {output_dir / filename}")


def save_predictions(df: pd.DataFrame, output_dir: Path, filename: str):
    """
    Saves a DataFrame of predictions to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    output_dir : Path
        The directory where the file will be saved.
    filename : str
        The name of the CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / filename, index=False)
    print(f"Predictions saved to {output_dir / filename}")


def save_model(model, output_dir: Path, filename: str):
    """
    Saves a trained model using pickle.

    Parameters
    ----------
    model : object
        The trained scikit-learn model to save.
    output_dir : Path
        The directory where the model will be saved.
    filename : str
        The name of the pickle file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {output_dir / filename}")


def check_and_load_existing_artifact(path: Path, artifact_type: str):
    """
    Checks if an artifact exists at the given path and loads it.

    Parameters
    ----------
    path : Path
        The path to the artifact file.
    artifact_type : str
        A string indicating the type of artifact (e.g., 'results', 'predictions', 'model').

    Returns
    -------
    object or None
        The loaded artifact if it exists, otherwise None.
    """
    if path.exists():
        print(f"Found existing {artifact_type} at {path}. Loading...")
        if artifact_type in ["results", "predictions"]:
            return pd.read_csv(path)
        elif artifact_type == "model":
            with open(path, "rb") as f:
                return pickle.load(f)
    print(f"No existing {artifact_type} found at {path}.")
    return None
