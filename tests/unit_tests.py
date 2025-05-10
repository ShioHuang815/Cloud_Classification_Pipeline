import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import requests
from requests.exceptions import RequestException

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.acquire_data import acquire_data
from modules.aws_util import upload_directory_to_s3
from modules.data_cleaning import clean_data, load_config
from modules.model_training import train_model
from pipeline import run_pipeline


class ExpandedPipelineTests(unittest.TestCase):
    test_count = 0
    passed_count = 0

    def setUp(self):
        """Set up test fixtures for each test."""
        self.config = {
            "acquisition": {
                "source": "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/taylor/cloud.data",
                "raw_data_path": "artifacts/clouds.data",
                "output_path": "data/clouds.csv",
            },
            "data": {
                "input_path": "data/clouds.csv",
                "output_path": "artifacts/cleaned_data.csv",
            },
            "model": {
                "features": [
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
                ],
                "target": "class",
                "test_size": 0.2,
                "random_state": 42,
                "model_path": "artifacts/model.pkl",
            },
            "metrics": {"output_path": "artifacts/metrics"},
            "aws": {
                "bucket_name": "cloud-classification-artifacts",
                "s3_folder": "cloud-classifier",
                "region_name": "us-east-1",
            },
        }

        # Test dataframe that matches expected format
        self.test_data = pd.DataFrame(
            {
                "visible_mean": [1.0, 2.0, 3.0, 4.0],
                "visible_max": [5.0, 6.0, 7.0, 8.0],
                "visible_min": [0.5, 1.5, 2.5, 3.5],
                "visible_mean_distribution": [0.1, 0.2, 0.3, 0.4],
                "visible_contrast": [0.5, 0.6, 0.7, 0.8],
                "visible_entropy": [1.1, 1.2, 1.3, 1.4],
                "visible_second_angular_momentum": [0.01, 0.02, 0.03, 0.04],
                "IR_mean": [10.0, 20.0, 30.0, 40.0],
                "IR_max": [15.0, 25.0, 35.0, 45.0],
                "IR_min": [5.0, 15.0, 25.0, 35.0],
                "class": [0, 1, 0, 1],
            }
        )

    @classmethod
    def setUpClass(cls):
        """Initialize counters before any tests run."""
        cls.test_count = 0
        cls.passed_count = 0

    def tearDown(self):
        """Count each test after it runs."""
        self.__class__.test_count += 1

    @classmethod
    def tearDownClass(cls):
        """Print final summary after all tests."""
        print("\n" + "=" * 60)
        print(f"TEST SUMMARY: {cls.passed_count} of {cls.test_count} tests passed")
        print("=" * 60)

    def print_success(self, message):
        """Helper to print success messages."""
        self.__class__.passed_count += 1
        print(f"\033[92m✓ SUCCESS: {message}\033[0m")  # Green color

    def print_failure(self, message):
        """Helper to print failure messages."""
        print(f"\033[91m✗ FAILED: {message}\033[0m")  # Red color

    # Happy Tests
    @patch("modules.data_cleaning.pd.read_csv")
    def test_clean_data_happy_path(self, mock_read_csv):
        """Happy Test: Data cleaning handles valid input correctly."""
        print("\n" + "=" * 60)
        print("TEST 1: Data cleaning with valid input (Happy Path)")
        print("=" * 60)

        # Setup mock
        mock_read_csv.return_value = self.test_data

        try:
            x_train, x_test, y_train, y_test = clean_data(self.config)

            # Verify shapes match expected test_size (20% of 4 samples = 1)
            self.assertEqual(len(x_train), 3)
            self.assertEqual(len(x_test), 1)
            self.assertEqual(len(y_train), 3)
            self.assertEqual(len(y_test), 1)

            self.print_success("Data cleaned and split correctly with expected shapes")
        except Exception as e:
            self.print_failure(f"Error during data cleaning: {str(e)}")
            raise

    @patch("modules.model_training.RandomForestClassifier")
    @patch("modules.model_training.joblib.dump")
    def test_model_training_happy_path(self, mock_dump, mock_rf):
        """Happy Test: Model training completes with valid data."""
        print("\n" + "=" * 60)
        print("TEST 2: Model training with valid data (Happy Path)")
        print("=" * 60)

        # Setup test data with all features from config
        X_train = self.test_data[self.config["model"]["features"]].iloc[:3]
        X_test = self.test_data[self.config["model"]["features"]].iloc[3:]
        y_train = self.test_data["class"].iloc[:3]
        y_test = self.test_data["class"].iloc[3:]

        # Mock the classifier
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        mock_model.predict = MagicMock(return_value=y_test.values)
        mock_rf.return_value = mock_model

        # Mock the model saving
        mock_dump.return_value = None

        try:
            model, accuracy = train_model(X_train, X_test, y_train, y_test, self.config)

            # Verify model was trained with correct parameters
            mock_model.fit.assert_called_once_with(X_train, y_train)
            self.assertIsNotNone(model)
            self.assertTrue(0 <= accuracy <= 1)

            self.print_success(
                f"Model trained successfully with accuracy: {accuracy:.2f}"
            )
        except Exception as e:
            self.print_failure(f"Error during model training: {str(e)}")
            raise

    @patch("modules.aws_util.boto3.client")
    def test_s3_upload_happy_path(self, mock_boto):
        """Happy Test: S3 upload completes successfully with config values."""
        print("\n" + "=" * 60)
        print("TEST 3: Successful S3 upload with config (Happy Path)")
        print("=" * 60)

        # Setup mock S3 client
        mock_client = MagicMock()
        mock_client.head_bucket = MagicMock()
        mock_client.upload_file = MagicMock()
        mock_boto.return_value = mock_client

        try:
            result = upload_directory_to_s3(
                local_directory="artifacts",
                bucket_name=self.config["aws"]["bucket_name"],
                s3_folder=self.config["aws"]["s3_folder"],
                config={
                    "aws_access_key_id": "test-key",
                    "aws_secret_access_key": "test-secret",
                    "region_name": self.config["aws"]["region_name"],
                },
            )

            # Verify upload was attempted with correct bucket and folder
            mock_client.upload_file.assert_called()
            self.assertTrue(result)

            self.print_success(
                f"Files uploaded to S3 bucket: {self.config['aws']['bucket_name']}"
            )
        except Exception as e:
            self.print_failure(f"Error during S3 upload: {str(e)}")
            raise

    # Unhappy Tests
    @patch("modules.acquire_data.requests.get")
    def test_acquire_data_invalid_source(self, mock_get):
        """Unhappy Test: Data acquisition fails with invalid source URL."""
        print("\n" + "=" * 60)
        print("TEST 4: Invalid data source URL (Unhappy Path)")
        print("=" * 60)

        # Make request fail
        mock_get.side_effect = RequestException("Invalid URL")

        try:
            with self.assertRaises(Exception):
                acquire_data(
                    self.config["acquisition"]["source"],
                    self.config["acquisition"]["output_path"],
                )
            self.print_success("Properly rejected invalid source URL as expected")
        except Exception as e:
            self.print_failure(f"Did not handle invalid URL correctly: {str(e)}")
            raise

    @patch("modules.data_cleaning.pd.read_csv")
    def test_clean_data_missing_features(self, mock_read_csv):
        """Unhappy Test: Data cleaning fails with missing features."""
        print("\n" + "=" * 60)
        print("TEST 5: Missing required features (Unhappy Path)")
        print("=" * 60)

        # Setup mock with missing IR features
        bad_data = self.test_data.drop(columns=["IR_mean", "IR_max", "IR_min"])
        mock_read_csv.return_value = bad_data

        try:
            with self.assertRaises(Exception):
                clean_data(self.config)
            self.print_success(
                "Properly rejected data with missing features as expected"
            )
        except Exception as e:
            self.print_failure(f"Did not handle missing features correctly: {str(e)}")
            raise

    @patch("modules.aws_util.boto3.client")
    def test_s3_upload_invalid_credentials(self, mock_boto):
        """Unhappy Test: S3 upload fails with invalid credentials."""
        print("\n" + "=" * 60)
        print("TEST 6: Invalid AWS credentials (Unhappy Path)")
        print("=" * 60)

        # Make S3 client fail
        mock_boto.side_effect = Exception("Invalid credentials")

        try:
            with self.assertRaises(Exception):
                upload_directory_to_s3(
                    local_directory="artifacts",
                    bucket_name=self.config["aws"]["bucket_name"],
                    config={
                        "aws_access_key_id": "invalid",
                        "aws_secret_access_key": "invalid",
                        "region_name": self.config["aws"]["region_name"],
                    },
                )
            self.print_success("Properly rejected invalid credentials as expected")
        except Exception as e:
            self.print_failure(f"Did not handle credential error correctly: {str(e)}")
            raise


if __name__ == "__main__":
    unittest.main()
