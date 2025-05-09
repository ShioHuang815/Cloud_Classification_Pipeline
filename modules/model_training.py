"""Model training and evaluation module."""
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


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
    model = RandomForestClassifier(random_state=config["model"]["random_state"])
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Save model
    joblib.dump(model, config["model"]["model_path"])

    # Save visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    plt.title(f"Accuracy: {accuracy:.2f}")
    plt.savefig(config["metrics"]["output_path"])
    plt.close()

    return model, accuracy
