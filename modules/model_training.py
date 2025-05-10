"""Model training and evaluation module."""
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_component(data, path, name):
    """Helper function to save different components to files."""
    try:
        output_path = Path(path) / f"{name}.csv"
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data.to_csv(output_path, index=False)
        else:
            # For non-dataframe objects (like predictions), convert to DataFrame first
            pd.DataFrame(data).to_csv(output_path, index=False)
        logger.info(f"Successfully saved {name} to {output_path}")
    except PermissionError as e:
        logger.error(f"Permission denied when trying to save {name}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error saving {name}: {e}")
        raise


def train_model(x_train, x_test, y_train, y_test, config):
    """Train and evaluate a Random Forest classifier.

    Args:
        x_train: Training features DataFrame
        x_test: Test features DataFrame
        y_train: Training labels Series
        y_test: Test labels Series
        config: Configuration dictionary

    Returns:
        tuple: (trained model, accuracy score)
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(config["metrics"]["output_path"])
        output_path.mkdir(parents=True, exist_ok=True)

        # Save raw data components with exception handling
        if "raw_data_path" in config["data"]:
            try:
                raw_data = pd.read_csv(config["data"]["raw_data_path"])
                save_component(raw_data, output_path, "raw_data")
            except FileNotFoundError as e:
                logger.error(f"Raw data file not found: {e}")
                raise
            except pd.errors.EmptyDataError as e:
                logger.error(f"Raw data file is empty: {e}")
                raise

        if "cleaned_data_path" in config["data"]:
            try:
                cleaned_data = pd.read_csv(config["data"]["cleaned_data_path"])
                save_component(cleaned_data, output_path, "cleaned_data")
            except FileNotFoundError as e:
                logger.error(f"Cleaned data file not found: {e}")
                raise
            except pd.errors.EmptyDataError as e:
                logger.error(f"Cleaned data file is empty: {e}")
                raise

        # Save features and labels
        save_component(x_train, output_path, "training_features")
        save_component(x_test, output_path, "test_features")
        save_component(y_train, output_path, "training_labels")
        save_component(y_test, output_path, "test_labels")

        # Train model
        model = RandomForestClassifier(random_state=config["model"]["random_state"])
        model.fit(x_train, y_train)

        # Save model with exception handling
        try:
            model_path = Path(config["model"]["model_path"])
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            logger.info(f"Model successfully saved to {model_path}")
        except (PermissionError, FileNotFoundError) as e:
            logger.error(f"Error saving model: {e}")
            raise

        # Make predictions and calculate metrics
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Save predictions and metrics
        save_component(y_pred, output_path, "predictions")

        metrics = {
            "accuracy": accuracy,
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

        try:
            pd.DataFrame.from_dict(metrics["classification_report"]).transpose().to_csv(
                output_path / "classification_report.csv"
            )
            pd.DataFrame({"accuracy": [accuracy]}).to_csv(output_path / "accuracy.csv")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            raise

        # Save visualization with exception handling
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay(cm).plot(ax=ax)
            plt.title(f"Accuracy: {accuracy:.2f}")
            plt.savefig(output_path / "confusion_matrix.png")
            plt.close()
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
            raise

        return model, accuracy

    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        raise
