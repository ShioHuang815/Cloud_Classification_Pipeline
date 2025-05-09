"""AWS utility functions for S3 operations."""
import logging
import os

import boto3
from botocore.exceptions import ClientError, NoCredentialsError


def initialize_s3_client(config):
    """Initialize and return an S3 client with proper credentials."""
    return boto3.client(
        "s3",
        aws_access_key_id=config.get("aws_access_key_id"),
        aws_secret_access_key=config.get("aws_secret_access_key"),
        region_name=config.get("region_name", "us-east-1"),
    )


def create_s3_bucket_if_not_exists(s3_client, bucket_name, region):
    """Create S3 bucket if it doesn't exist."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logging.info("Bucket %s already exists", bucket_name)
        return False
    except ClientError as error:
        error_code = error.response["Error"]["Code"]
        if error_code == "404":
            try:
                if region == "us-east-1":
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": region},
                    )
                logging.info("Created bucket %s in region %s", bucket_name, region)
                return True
            except ClientError as create_error:
                logging.error(
                    "Failed to create bucket %s: %s", bucket_name, create_error
                )
                raise
        else:
            logging.error("Error accessing bucket %s: %s", bucket_name, error)
            raise


def upload_directory_to_s3(local_directory, bucket_name, s3_folder=None, config=None):
    """
    Upload a directory to S3 bucket, creating the bucket if it doesn't exist.

    Args:
        local_directory: Local directory to upload
        bucket_name: S3 bucket name
        s3_folder: Folder path in S3 (optional)
        config: AWS configuration dict
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not config:
        config = {}

    try:
        s3_client = initialize_s3_client(config)
        region = config.get("region_name", "us-east-1")
        create_s3_bucket_if_not_exists(s3_client, bucket_name, region)

        for root, _, files in os.walk(local_directory):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_directory)
                s3_path = (
                    os.path.join(s3_folder, relative_path)
                    if s3_folder
                    else relative_path
                )
                s3_path = s3_path.replace("\\", "/")

                try:
                    s3_client.upload_file(local_path, bucket_name, s3_path)
                    logger.info(
                        "Uploaded %s to s3://%s/%s", local_path, bucket_name, s3_path
                    )
                except ClientError as upload_error:
                    logger.error("Failed to upload %s: %s", local_path, upload_error)
                    raise

        logger.info("All files uploaded successfully")
        return True

    except NoCredentialsError:
        logger.error("AWS credentials not available")
        raise
    except Exception as error:
        logger.error("Error uploading to S3: %s", error)
        raise
    