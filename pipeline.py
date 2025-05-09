"""Cloud classification pipeline main module.

This module orchestrates the complete workflow including:
- Data loading and cleaning
- Model training
- Artifact generation
- Optional S3 upload
"""
import logging
import os
from typing import Tuple

import pandas as pd
from modules.aws_util import upload_directory_to_s3
from modules.data_cleaning import clean_data, load_config
from modules.model_training import train_model


def run_pipeline() -> None:
    """Execute the complete cloud classification pipeline."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        logger.info("Loading configuration...")
        config = load_config()

        logger.info("Starting data cleaning...")
        x_train, x_test, y_train, y_test = clean_data(config)

        logger.info("Starting model training...")
        trained_model, accuracy = train_model(x_train, x_test, y_train, y_test, config)
        logger.info("Model trained with accuracy: %.2f", accuracy)

        if "aws" in config:
            logger.info("Preparing to upload artifacts to S3...")
            bucket_name = os.getenv("AWS_BUCKET_NAME", config["aws"]["bucket_name"])

            if bucket_name:
                aws_config = {
                    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                    "region_name": config["aws"]["region_name"],
                }

                upload_directory_to_s3(
                    local_directory="artifacts",
                    bucket_name=bucket_name,
                    s3_folder=config["aws"].get("s3_folder"),
                    config=aws_config,
                )
                logger.info("Artifacts successfully uploaded to S3")
            else:
                logger.warning("No S3 bucket configured. Skipping upload.")
        else:
            logger.warning("No AWS configuration found. Skipping upload.")

    except Exception as error:
        logger.error("Pipeline failed: %s", error)
        raise


if __name__ == "__main__":
    run_pipeline()
    