"""
This module provides utilities for testing the CNN model on new transformer data.
It includes functions to load pre-trained models and make predictions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import joblib

# Define paths
OUTPUT_PATH = "../../outputs"
MODEL_PATH = os.path.join(OUTPUT_PATH, "models")
PREDICTION_PATH = os.path.join(OUTPUT_PATH, "predictions")

# Create output directory if it doesn't exist
os.makedirs(PREDICTION_PATH, exist_ok=True)


def load_and_prepare_data(data_path, scaler_path=None, scale_data=True):
    """
    Load and prepare data for prediction.

    Args:
        data_path (str): Path to CSV file with transformer data.
        scaler_path (str, optional): Path to saved scaler. Defaults to None.
        scale_data (bool): Whether to scale the data or not.

    Returns:
        pd.DataFrame: Prepared data ready for prediction.
    """
    # Load data
    df = pd.read_csv(data_path)

    # Extract features (assuming Health_index is not in the input data for prediction)
    if 'Health_index' in df.columns:
        features = df.drop(columns=['Health_index'])
        has_target = True
        targets = df['Health_index']
    else:
        features = df
        has_target = False

    # Scale features if needed
    if scale_data and scaler_path:
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            features_scaled = pd.DataFrame(
                scaler.transform(features),
                columns=features.columns
            )
        else:
            print(f"Warning: Scaler file {scaler_path} not found. Using unscaled data.")
            features_scaled = features
    else:
        features_scaled = features

    return features_scaled, has_target, targets if has_target else None


def predict_with_cnn(data, model_path=None):
    """
    Make predictions using the trained CNN model.

    Args:
        data (pd.DataFrame): Prepared data for prediction.
        model_path (str): Path to the saved CNN model.

    Returns:
        np.ndarray: Predicted health index values.
    """
    # Set default model path if none provided
    if model_path is None:
        model_path = os.path.join(MODEL_PATH, "cnn_model_best.keras")

    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")

    # Load model
    try:
        model = keras.saving.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        try:
            model = keras.models.load_model(model_path)
        except Exception as e2:
            raise Exception(f"Failed to load model: {e2}")

    # Reshape data for CNN (samples, timesteps, features)
    X = data.values
    X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)

    # Make predictions
    try:
        predictions = model.predict(X_reshaped).flatten()
    except Exception as e:
        raise Exception(f"Error making predictions: {e}")

    return predictions


def interpret_health_index(health_index):
    """
    Interpret the health index value into a category and provide recommendations.

    Args:
        health_index (float): Predicted health index.

    Returns:
        tuple: Category (str), Description (str), Recommendation (str)
    """
    if health_index >= 85:
        category = "A - Minor Deterioration"
        description = "Excellent condition with minimal wear and tear."
        recommendation = "Continue routine maintenance like oil changes and inspections."
    elif health_index >= 70:
        category = "B - Significant Deterioration"
        description = "Good condition but with noticeable decline."
        recommendation = "Closer monitoring recommended. Consider corrective maintenance like tightening connections or addressing minor leaks."
    elif health_index >= 50:
        category = "C - Widespread Deterioration"
        description = "Significant degradation across various components with reduced lifespan."
        recommendation = "Preventative maintenance or refurbishment needed to avoid unexpected failures."
    elif health_index >= 30:
        category = "D - Widespread Very Serious Deterioration"
        description = "Critical stage with severe degradation throughout the system."
        recommendation = "Immediate action required to prevent catastrophic failure. Consider replacement or emergency repairs."
    else:
        category = "E - Extensive Deterioration"
        description = "Extensive damage, no longer operational."
        recommendation = "Replacement is the only viable course of action."

    return category, description, recommendation


def generate_report(data, predictions, actual=None, output_path=None):
    """
    Generate a comprehensive report of the predictions.

    Args:
        data (pd.DataFrame): Original input data.
        predictions (np.ndarray): Predicted health index values.
        actual (np.ndarray, optional): Actual health index values if available.
        output_path (str): Path to save the report.
    """
    # Set default output path if none provided
    if output_path is None:
        output_path = PREDICTION_PATH

    # Create results DataFrame
    results = data.copy()
    results['Predicted_Health_Index'] = predictions

    if actual is not None:
        results['Actual_Health_Index'] = actual
        results['Error'] = actual - predictions

    # Add interpretations
    categories = []
    descriptions = []
    recommendations = []

    for pred in predictions:
        category, description, recommendation = interpret_health_index(pred)
        categories.append(category)
        descriptions.append(description)
        recommendations.append(recommendation)

    results['Category'] = categories
    results['Description'] = descriptions
    results['Recommendation'] = recommendations

    # Save to CSV
    results.to_csv(os.path.join(output_path, "prediction_results.csv"), index=False)

    # If we have actual values, create a comparison plot
    if actual is not None:
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predictions, alpha=0.7)

        # Add perfect prediction line
        min_val = min(min(actual), min(predictions))
        max_val = max(max(actual), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.title('Actual vs. Predicted Health Index')
        plt.xlabel('Actual Health Index')
        plt.ylabel('Predicted Health Index')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "prediction_comparison.png"))
        plt.close()

    # Create category distribution plot
    category_counts = results['Category'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    bars = plt.bar(category_counts.index, category_counts.values)

    # Add count labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.1,
            str(int(height)),
            ha='center',
            va='bottom'
        )

    plt.title('Distribution of Transformer Health Categories')
    plt.xlabel('Health Category')
    plt.ylabel('Count')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "category_distribution.png"))
    plt.close()

    # Create a simple health index distribution
    plt.figure(figsize=(12, 6))
    plt.hist(predictions, bins=20, alpha=0.7, color='blue')
    if actual is not None:
        plt.hist(actual, bins=20, alpha=0.5, color='green')
        plt.legend(['Predicted', 'Actual'])
    else:
        plt.legend(['Predicted'])

    plt.title('Distribution of Health Index Values')
    plt.xlabel('Health Index')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "health_index_distribution.png"))
    plt.close()

    print(f"Report generated and saved to {output_path}")


def sample_data_if_missing(data_path=None):
    """
    Create a sample data file if none is provided.

    Args:
        data_path (str, optional): Path to save the sample data. Defaults to None.

    Returns:
        str: Path to the sample data file.
    """
    if data_path is None:
        data_path = os.path.join(OUTPUT_PATH, "sample_transformers.csv")

    # Create sample data with the same format as the original dataset
    sample_data = {
        'Hydrogen': [10, 20, 30, 40, 50],
        'Oxigen': [500, 600, 700, 800, 900],
        'Nitrogen': [50000, 55000, 60000, 65000, 70000],
        'Methane': [100, 200, 300, 400, 500],
        'CO': [200, 300, 400, 500, 600],
        'CO2': [1500, 2000, 2500, 3000, 3500],
        'Ethylene': [50, 100, 150, 200, 250],
        'Ethane': [100, 200, 300, 400, 500],
        'Acethylene': [10, 20, 30, 40, 50],
        'DBDS': [100, 120, 140, 160, 180],
        'Power_factor': [1.0, 1.5, 2.0, 2.5, 3.0],
        'Interfacial_V': [40, 42, 44, 46, 48],
        'Dielectric_rigidity': [52, 54, 56, 58, 60],
        'Water_content': [5, 10, 15, 20, 25]
    }

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(sample_data)
    df.to_csv(data_path, index=False)

    print(f"Created sample data file at {data_path}")
    return data_path


def main(data_path=None, model_path=None, scaler_path=None, create_sample=False):
    """
    Main function to load data, make predictions, and generate a report.

    Args:
        data_path (str): Path to CSV file with transformer data.
        model_path (str): Path to saved model.
        scaler_path (str): Path to saved scaler.
        create_sample (bool): Whether to create a sample data file if none is provided.
    """
    # Set default paths if none provided
    if model_path is None:
        model_path = os.path.join(MODEL_PATH, "cnn_model_best.keras")
    if scaler_path is None:
        scaler_path = os.path.join(MODEL_PATH, "scaler.pkl")

    # If no data path is provided and create_sample is True, create a sample data file
    if data_path is None and create_sample:
        data_path = sample_data_if_missing()
    elif data_path is None:
        print("Error: No data path provided. Please specify a path to a CSV file with transformer data.")
        print("You can also run with --create_sample to create a sample data file.")
        return

    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    try:
        print("Loading and preparing data...")
        features, has_target, targets = load_and_prepare_data(
            data_path, scaler_path, scale_data=True
        )

        print("Making predictions...")
        predictions = predict_with_cnn(features, model_path)

        print("Generating report...")
        if has_target:
            generate_report(features, predictions, targets)

            # Calculate metrics
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            mse = mean_squared_error(targets, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets, predictions)
            r2 = r2_score(targets, predictions)

            print("\nModel Performance on This Dataset:")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"R-squared (R²): {r2:.4f}")
        else:
            generate_report(features, predictions)

        print("\nPrediction complete!")
        print(f"Results saved to {PREDICTION_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Predict transformer health index using CNN model')
    parser.add_argument('--data_path', type=str, help='Path to CSV file with transformer data')
    parser.add_argument('--model_path', type=str, default=None, help='Path to saved model')
    parser.add_argument('--scaler_path', type=str, default=None, help='Path to saved scaler')
    parser.add_argument('--create_sample', action='store_true', help='Create a sample data file if none is provided')

    args = parser.parse_args()

    main(args.data_path, args.model_path, args.scaler_path, args.create_sample)