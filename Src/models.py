from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
import numpy as np

def train_model(X, y):
    """Train a predictive model."""
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = RandomForestClassifier()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

# Example usage
if __name__ == "_main_":
    # Generating a random dataset for demonstration purposes
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.randint(2, size=100)  # Binary target

    model, accuracy = train_model(X, y)
    print(f"Model accuracy: {accuracy}")