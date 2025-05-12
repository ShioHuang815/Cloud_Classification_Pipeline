# Paste pre-Pyrfected Python
"""Cloud classification pipeline main module.

This module orchestrates the complete workflow including:
- Data acquisition
- Data loading and cleaning
- Feature engineering
- Model training
- Artifact generation
- Optional S3 upload
"""

import logging
import os
from typing import Tuple

import pandas as pd
from modules.acquire_data import acquire_data
from modules.aws_util import upload_directory_to_s3
from modules.data_cleaning import clean_data, load_config
from modules.feature_engineering import generate_features
from modules.model_training import train_model


def run_pipeline() -> None:
    """Execute the complete cloud classification pipeline."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        logger.info("Loading configuration...")
        config = load_config()

        # Prepare directories
        raw_data_path = os.path.join("artifacts", "clouds.data")
        os.makedirs("artifacts", exist_ok=True)
        os.makedirs(
            os.path.dirname(config["acquisition"]["output_path"]), exist_ok=True
        )
        os.makedirs(os.path.dirname(config["model"]["model_path"]), exist_ok=True)

        # Data acquisition
        logger.info("Starting data acquisition...")
        try:
            df = acquire_data(
                url=config["acquisition"]["source"],
                csv_path=config["acquisition"]["output_path"],
            )
            logger.info(
                f"Processed data saved to {config['acquisition']['output_path']}"
            )
        except Exception as e:
            logger.error(f"Failed to acquire data: {e}")
            raise

        # Data loading and cleaning
        logger.info("Starting data loading and cleaning...")
        try:
            train_data, test_data = clean_data(config)
            all_data = pd.concat([train_data, test_data], ignore_index=True)
            all_data.reset_index(drop=True, inplace=True)
        except Exception as e:
            logger.error(f"Failed to clean data: {e}")
            raise

        # Feature Engineering
        logger.info("Starting feature engineering...")
        try:
            all_data = generate_features(all_data)

            # Split back into train and test
            train_data = all_data.iloc[: len(train_data)].copy()
            test_data = all_data.iloc[len(train_data) :].copy()
        except Exception as e:
            logger.error(f"Failed to generate features: {e}")
            raise

        # Model training
        logger.info("Starting model training...")
        try:
            # Dynamically add new features
            additional_features = [
                "log_entropy",
                "entropy_x_contrast",
                "IR_range",
                "IR_norm_range",
            ]
            features = config["model"]["features"] + additional_features

            # Validate feature existence
            missing_features = [f for f in features if f not in train_data.columns]
            if missing_features:
                logger.error(f"Missing features in the dataset: {missing_features}")
                raise ValueError(f"Missing features: {missing_features}")

            x_train = train_data[features]
            y_train = train_data[config["model"]["target"]]
            x_test = test_data[features]
            y_test = test_data[config["model"]["target"]]

            trained_model, accuracy = train_model(
                x_train, x_test, y_train, y_test, config
            )
            logger.info(f"Model trained with accuracy: {accuracy:.2f}")
        except Exception as error:
            logger.error(f"Model training failed: {error}")
            raise

        # Optional S3 upload
        if "aws" in config:
            logger.info("Preparing to upload artifacts to S3...")
            bucket_name = os.getenv("AWS_BUCKET_NAME", config["aws"].get("bucket_name"))

            if bucket_name:
                aws_config = {
                    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                    "region_name": config["aws"]["region_name"],
                }

                try:
                    upload_directory_to_s3(
                        local_directory="artifacts",
                        bucket_name=bucket_name,
                        s3_folder=config["aws"].get("s3_folder", ""),
                        config=aws_config,
                    )
                    logger.info("Artifacts successfully uploaded to S3")
                except Exception as e:
                    logger.error(f"Failed to upload artifacts to S3: {e}")
                    raise
            else:
                logger.warning("No S3 bucket configured. Skipping upload.")
        else:
            logger.warning("No AWS configuration found. Skipping upload.")

    except Exception as error:
        logger.error(f"Pipeline failed: {error}")
        raise


if __name__ == "__main__":
    run_pipeline()
