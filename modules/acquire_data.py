"""Module for acquiring and preprocessing the cloud dataset from a URL.

This includes:
- Downloading the raw `clouds.data` file via HTTP
- Parsing and converting string values to numeric types
- Labeling each observation with its class
- Saving the processed data to CSV and raw data to file
"""

import logging
import os

import pandas as pd
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Column names for the original dataset
columns = [
    "visible_mean",
    "visible_max",
    "visible_min",
    "visible_mean_distribution",
    "visible_contrast",
    "visible_entropy",
    "visible_second_angular_momentum",
    "IR_mean",
    "IR_max",
    "IR_min",
]


def acquire_data(url: str, csv_path: str) -> pd.DataFrame:
    """
    Acquire and preprocess the cloud data from a URL.

    Downloads the raw data file, cleans and splits text, converts all values to floats,
    assigns class labels (0 for first cloud, 1 for second cloud), and saves
    the result to a CSV file.

    Parameters
    ----------
    url : str
        HTTP URL to the raw `clouds.data` file.
    csv_path : str
        The path where the processed DataFrame will be saved as CSV.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with original features and a `'class'` column.
    """
    try:
        # Fetch the raw text via HTTP
        resp = requests.get(url)
        resp.raise_for_status()
        lines = resp.text.splitlines()

        # Split each line on whitespace
        raw = [line.split() for line in lines]

        # Process first cloud samples (class 0)
        first_raw = raw[53:1077]
        first = pd.DataFrame(
            [[float(s) for s in row] for row in first_raw], columns=columns
        )
        first["class"] = 0

        # Process second cloud samples (class 1)
        second_raw = raw[1082:2105]
        second = pd.DataFrame(
            [[float(s) for s in row] for row in second_raw], columns=columns
        )
        second["class"] = 1

        # Combine and reset index
        data = pd.concat([first, second], ignore_index=True)

        # Ensure directory exists and save to CSV
        try:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            data.to_csv(csv_path, index=False)
            logger.info("Successfully saved processed data to %s", csv_path)
        except Exception as e:
            logger.error("Failed to save processed data to %s: %s", csv_path, e)
            raise

        return data

    except Exception as e:
        logger.error("Data acquisition failed: %s", e)
        raise
