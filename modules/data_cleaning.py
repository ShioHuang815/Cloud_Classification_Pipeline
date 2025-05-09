"""Data cleaning and preparation module."""
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Dictionary of configuration parameters
    """
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return config


def clean_data(config):
    """Clean and prepare data for modeling.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (x_train, x_test, y_train, y_test)
    """
    data_frame = pd.read_csv(config["data"]["input_path"])
    data_frame = data_frame.drop_duplicates().dropna()

    features = data_frame[config["model"]["features"]]
    target = data_frame[config["model"]["target"]]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
    )

    os.makedirs(os.path.dirname(config["data"]["output_path"]), exist_ok=True)
    train_data = pd.concat([x_train, y_train], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1)

    train_data.to_csv(
        config["data"]["output_path"].replace(".csv", "_train.csv"), index=False
    )
    test_data.to_csv(
        config["data"]["output_path"].replace(".csv", "_test.csv"), index=False
    )

    return x_train, x_test, y_train, y_test
