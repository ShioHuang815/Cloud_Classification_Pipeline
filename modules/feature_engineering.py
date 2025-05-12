"""
Module for feature engineering.
"""

import numpy as np
import pandas as pd


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates additional features from the input DataFrame.

    Args:
        df: Input DataFrame containing the original features.

    Returns:
        DataFrame with the added features: 'log_entropy', 'entropy_x_contrast',
        'IR_range', and 'IR_norm_range'.
    """
    df["log_entropy"] = df["visible_entropy"].apply(np.log)
    df["entropy_x_contrast"] = df["visible_contrast"] * df["visible_entropy"]
    df["IR_range"] = df["IR_max"] - df["IR_min"]
    df["IR_norm_range"] = df["IR_range"] / df["IR_mean"]
    return df
