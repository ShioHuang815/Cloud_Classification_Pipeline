# tests/unit_test.py
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import yaml

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.data_cleaning import clean_data, load_config
from modules.model_training import train_model


class TestDataCleaning(unittest.TestCase):
    def setUp(self):
        self.config = {
            "data": {
                "input_path": "data/test_data.csv",
                "output_path": "artifacts/cleaned_data.csv",
            },
            "model": {
                "features": ["feature1", "feature2"],
                "target": "target",
                "test_size": 0.2,
                "random_state": 42,
            },
        }

        self.test_data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "target": [0, 1, 0, 1, 0],
            }
        )

    def test_clean_data_happy_path(self):
        """Happy Test: clean_data works with valid input"""
        print("\nHappy Test 1: clean_data with valid input - ", end="")
        self.test_data.to_csv(self.config["data"]["input_path"], index=False)

        try:
            X_train, X_test, y_train, y_test = clean_data(self.config)
            self.assertIsNotNone(X_train)
            self.assertIsNotNone(X_test)
            self.assertEqual(len(X_train) + len(X_test), len(self.test_data))
            self.assertEqual(len(y_train) + len(y_test), len(self.test_data))
            print("SUCCESS - Correctly processed valid data")
        except Exception as e:
            print(f"FAILED - Error with valid data: {str(e)}")
            raise
        finally:
            if os.path.exists(self.config["data"]["input_path"]):
                os.remove(self.config["data"]["input_path"])

    def test_clean_data_missing_features(self):
        """Unhappy Test: clean_data handles missing features"""
        print("\nUnhappy Test 1: clean_data with missing features - ", end="")
        bad_data = pd.DataFrame({"wrong_feature": [1, 2, 3], "target": [0, 1, 0]})
        bad_data.to_csv(self.config["data"]["input_path"], index=False)

        try:
            with self.assertRaises(KeyError):
                clean_data(self.config)
            print("SUCCESS - Correctly detected missing features")
        except Exception as e:
            print(f"FAILED - Did not handle missing features correctly: {str(e)}")
            raise
        finally:
            if os.path.exists(self.config["data"]["input_path"]):
                os.remove(self.config["data"]["input_path"])

    def test_clean_data_empty_file(self):
        """Unhappy Test: clean_data handles empty file"""
        print("\nUnhappy Test 2: clean_data with empty file - ", end="")
        open(self.config["data"]["input_path"], "w").close()

        try:
            with self.assertRaises(pd.errors.EmptyDataError):
                clean_data(self.config)
            print("SUCCESS - Correctly detected empty file")
        except Exception as e:
            print(f"FAILED - Did not handle empty file correctly: {str(e)}")
            raise
        finally:
            if os.path.exists(self.config["data"]["input_path"]):
                os.remove(self.config["data"]["input_path"])

    def test_clean_data_missing_values(self):
        """Happy Test: clean_data handles missing values"""
        print("\nHappy Test 2: clean_data with missing values - ", end="")
        missing_data = self.test_data.copy()
        missing_data.loc[0, "feature1"] = np.nan
        missing_data.to_csv(self.config["data"]["input_path"], index=False)

        try:
            X_train, X_test, y_train, y_test = clean_data(self.config)
            self.assertEqual(len(X_train) + len(X_test), len(missing_data.dropna()))
            print("SUCCESS - Correctly handled missing values")
        except Exception as e:
            print(f"FAILED - Error handling missing values: {str(e)}")
            raise
        finally:
            if os.path.exists(self.config["data"]["input_path"]):
                os.remove(self.config["data"]["input_path"])

    def test_clean_data_duplicates(self):
        """Happy Test: clean_data removes duplicates"""
        print("\nHappy Test 3: clean_data with duplicates - ", end="")
        duplicate_data = pd.concat([self.test_data, self.test_data.iloc[[0]]])
        duplicate_data.to_csv(self.config["data"]["input_path"], index=False)

        try:
            X_train, X_test, y_train, y_test = clean_data(self.config)
            self.assertEqual(len(X_train) + len(X_test), len(self.test_data))
            print("SUCCESS - Correctly removed duplicates")
        except Exception as e:
            print(f"FAILED - Error handling duplicates: {str(e)}")
            raise
        finally:
            if os.path.exists(self.config["data"]["input_path"]):
                os.remove(self.config["data"]["input_path"])


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.config = {
            "model": {"random_state": 42, "model_path": "artifacts/model.pkl"},
            "metrics": {"output_path": "artifacts/metrics.png"},
        }

        self.X_train = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [10, 20, 30, 40, 50]}
        )
        self.X_test = pd.DataFrame({"feature1": [6, 7], "feature2": [60, 70]})
        self.y_train = pd.Series([0, 1, 0, 1, 0])
        self.y_test = pd.Series([1, 0])

    def test_train_model_happy_path(self):
        """Happy Test: model trains with valid input"""
        print("\nHappy Test 4: train_model with valid input - ", end="")
        try:
            model, accuracy = train_model(
                self.X_train, self.X_test, self.y_train, self.y_test, self.config
            )
            self.assertIsNotNone(model)
            self.assertTrue(0 <= accuracy <= 1)
            print("SUCCESS - Model trained successfully")
        except Exception as e:
            print(f"FAILED - Error training model: {str(e)}")
            raise

    def test_train_model_empty_data(self):
        """Unhappy Test: model training handles empty data"""
        print("\nUnhappy Test 3: train_model with empty data - ", end="")
        try:
            with self.assertRaises(ValueError):
                train_model(
                    pd.DataFrame(),
                    self.X_test,
                    pd.Series(dtype="float"),
                    self.y_test,
                    self.config,
                )
            print("SUCCESS - Correctly rejected empty data")
        except Exception as e:
            print(f"FAILED - Did not handle empty data correctly: {str(e)}")
            raise

    def test_train_model_mismatched_lengths(self):
        """Unhappy Test: model training handles mismatched lengths"""
        print("\nUnhappy Test 4: train_model with mismatched lengths - ", end="")
        try:
            with self.assertRaises(ValueError):
                train_model(
                    self.X_train,
                    self.X_test,
                    self.y_train.iloc[:-1],
                    self.y_test,
                    self.config,
                )
            print("SUCCESS - Correctly detected length mismatch")
        except Exception as e:
            print(f"FAILED - Did not handle length mismatch correctly: {str(e)}")
            raise

    @patch("joblib.dump")
    def test_train_model_saves_model(self, mock_dump):
        """Happy Test: model saves correctly"""
        print("\nHappy Test 5: train_model model saving - ", end="")
        try:
            train_model(
                self.X_train, self.X_test, self.y_train, self.y_test, self.config
            )
            mock_dump.assert_called_once()
            print("SUCCESS - Model saved correctly")
        except Exception as e:
            print(f"FAILED - Model save failed: {str(e)}")
            raise

    @patch("matplotlib.pyplot.savefig")
    def test_train_model_saves_metrics(self, mock_savefig):
        """Happy Test: metrics are saved correctly"""
        print("\nHappy Test 6: train_model metrics saving - ", end="")
        try:
            train_model(
                self.X_train, self.X_test, self.y_train, self.y_test, self.config
            )
            mock_savefig.assert_called_once()
            print("SUCCESS - Metrics saved correctly")
        except Exception as e:
            print(f"FAILED - Metrics save failed: {str(e)}")
            raise


if __name__ == "__main__":
    unittest.main()
    