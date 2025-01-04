
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import joblib
import numpy as np

def train_linear_regression(df, target_column, test_size=0.2):
    """
    Train a linear regression model on the provided DataFrame.

    Args:
    - df (pd.DataFrame): The data to train the model on.
    - target_column (str): The column name for the target variable.
    - test_size (float): The proportion of data to be used for testing (default is 0.2).

    Returns:
    - model (sklearn.linear_model.LinearRegression): The trained linear regression model.
    - X_test (pd.DataFrame): Test data features.
    - y_test (pd.Series): Test data target variable.
    """
    # Split the data into features and target variable
    X = df.drop(columns=[target_column])  # All columns except the target
    y = df[target_column]  # The target column

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Display the shape and first 5 rows of the train and test data
    print("Training Data:")
    print(f"Shape: {X_train.shape}")
    print(f"First 5 rows:\n{X_train.head()}")

    print("\nTest Data:")
    print(f"Shape: {X_test.shape}")
    print(f"First 5 rows:\n{X_test.head()}")

    print("\nTraining Target:")
    print(f"Shape: {y_train.shape}")
    print(f"First 5 rows:\n{y_train.head()}")

    print("\nTest Target:")
    print(f"Shape: {y_test.shape}")
    print(f"First 5 rows:\n{y_test.head()}")

    # Initialize the model
    model = LinearRegression()

    # K-Fold Cross Validation
    kfold_validation = KFold(4)
    cross_val_results = cross_val_score(model, X_train, y_train, cv=kfold_validation)
    print(f"K-Fold Cross Validation Results: {cross_val_results}")
    print(f"Mean Cross Validation Score: {np.mean(cross_val_results)}")

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Linear Regression Model Performance:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

    # Save the model to the models directory
    model_path = "../../outputs/models/linear_regression_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Prepare the input data for prediction
    Model_Input = {
        'Hydrogen': 15,
        'Oxigen': 308,
        'Nitrogen': 39700,
        'Methane': 3,
        'CO': 64,
        'CO2': 581,
        'Ethylene': 5,
        'Ethane': 27,
        'Acethylene': 0,
        'DBDS': 500,
        'Power_factor': 1,
        'Interfacial_V': 32,
        'Dielectric_rigidity': 60,
        'Water_content': 18,
        'Life_expectation': 51,
    }

    input_data = pd.DataFrame([Model_Input])
    Final = pd.concat([df, input_data], ignore_index=True)

    x = Final.drop(["Health_index"], axis=1)[:447]
    y = Final[['Health_index']][:447]
    x_Final = Final.drop(["Health_index"], axis=1)[447:]

    # Fit the model again and make predictions for the new input
    model.fit(x, y)
    predicted_cluster = model.predict(x_Final)

    print(f"Predicted Cluster Values: {predicted_cluster}")

    return model, X_test, y_test, y_pred


def main():
    # Load the preprocessed scaled data (scaled_data.csv)
    df = pd.read_csv("../../outputs/models/scaled_data.csv")

    # Target column for prediction
    target_column = "Health_index"  # You can change this if needed

    # Train the model
    model, X_test, y_test, y_pred = train_linear_regression(df, target_column)

    # Evaluate the model performance (optional)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Final Evaluation Performance:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")


if __name__ == "__main__":
    main()
