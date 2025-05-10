"""Model training and evaluation module."""
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


def save_component(data, path, name):
    """Helper function to save different components to files."""
    output_path = Path(path) / f"{name}.csv"
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data.to_csv(output_path, index=False)
    else:
        # For non-dataframe objects (like predictions), convert to DataFrame first
        pd.DataFrame(data).to_csv(output_path, index=False)


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
    # Create output directory if it doesn't exist
    output_path = Path(config["metrics"]["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Save raw data components (assuming they're passed in config)
    if "raw_data_path" in config["data"]:
        raw_data = pd.read_csv(config["data"]["raw_data_path"])
        save_component(raw_data, output_path, "raw_data")

    if "cleaned_data_path" in config["data"]:
        cleaned_data = pd.read_csv(config["data"]["cleaned_data_path"])
        save_component(cleaned_data, output_path, "cleaned_data")

    # Save features
    save_component(x_train, output_path, "training_features")
    save_component(x_test, output_path, "test_features")

    # Save train/test labels
    save_component(y_train, output_path, "training_labels")
    save_component(y_test, output_path, "test_labels")

    # Train model (TMO)
    model = RandomForestClassifier(random_state=config["model"]["random_state"])
    model.fit(x_train, y_train)

    # Save model
    model_path = Path(config["model"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    # Make predictions
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Save predictions
    save_component(y_pred, output_path, "predictions")

    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }
    pd.DataFrame.from_dict(metrics["classification_report"]).transpose().to_csv(
        output_path / "classification_report.csv"
    )
    pd.DataFrame({"accuracy": [accuracy]}).to_csv(output_path / "accuracy.csv")

    # Save visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    plt.title(f"Accuracy: {accuracy:.2f}")
    plt.savefig(output_path / "confusion_matrix.png")
    plt.close()

    return model, accuracy
