"""
Metrics and visualization utilities for evaluating the CNN model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    confusion_matrix,
    classification_report,
    accuracy_score
)
import keras
import joblib

# Define paths
OUTPUT_PATH = "../../outputs"
MODEL_PATH = os.path.join(OUTPUT_PATH, "models")
EVAL_PATH = os.path.join(OUTPUT_PATH, "graphs", "cnn_evaluation")

# Create output directory if it doesn't exist
os.makedirs(EVAL_PATH, exist_ok=True)


def classify_health_index(value):
    """
    Convert health index values to categorical classes.

    Args:
        value (float): Health index value.

    Returns:
        str: Health index category.
    """
    if value >= 85:
        return "A"
    elif value >= 70:
        return "B"
    elif value >= 50:
        return "C"
    elif value >= 30:
        return "D"
    else:
        return "E"


def evaluate_model_metrics(y_true, y_pred, output_path=None):
    """
    Calculate and visualize performance metrics for regression.

    Args:
        y_true (np.ndarray): True health index values.
        y_pred (np.ndarray): Predicted health index values.
        output_path (str): Path to save the visualizations.

    Returns:
        dict: Dictionary of metrics.
    """
    # Set default output path if none provided
    if output_path is None:
        output_path = EVAL_PATH

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calculate additional metrics
    max_error = np.max(np.abs(y_true - y_pred))
    mean_error = np.mean(y_true - y_pred)
    std_error = np.std(y_true - y_pred)

    # Store metrics in dictionary
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Max Error': max_error,
        'Mean Error': mean_error,
        'Std Error': std_error
    }

    # Create metrics visualization
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        ['MSE', 'RMSE', 'MAE', 'Max Error', 'Mean Error', 'Std Error'],
        [mse, rmse, mae, max_error, abs(mean_error), std_error]
    )

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.1,
            f"{height:.2f}",
            ha='center',
            va='bottom'
        )

    plt.title('Regression Error Metrics')
    plt.ylabel('Error Value')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "regression_metrics.png"))
    plt.close()

    # Create R² visualization
    plt.figure(figsize=(8, 6))
    plt.bar(['R²'], [r2], color='green')
    plt.axhline(y=0.7, color='red', linestyle='--', label='Good fit threshold (0.7)')
    plt.axhline(y=0.9, color='blue', linestyle='--', label='Excellent fit threshold (0.9)')
    plt.title('R-squared (R²) Score')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y')
    plt.text(0, r2 + 0.02, f"{r2:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "r_squared.png"))
    plt.close()

    return metrics


def evaluate_categorical_metrics(y_true, y_pred, output_path=None):
    """
    Calculate and visualize performance metrics when treating predictions as categories.

    Args:
        y_true (np.ndarray): True health index values.
        y_pred (np.ndarray): Predicted health index values.
        output_path (str): Path to save the visualizations.

    Returns:
        dict: Dictionary of categorical metrics.
    """
    # Set default output path if none provided
    if output_path is None:
        output_path = EVAL_PATH

    # Convert numeric health index to categories
    y_true_cat = [classify_health_index(val) for val in y_true]
    y_pred_cat = [classify_health_index(val) for val in y_pred]

    # Find all unique categories in both true and predicted
    unique_categories = sorted(set(y_true_cat) | set(y_pred_cat))

    # Calculate metrics
    accuracy = accuracy_score(y_true_cat, y_pred_cat)

    # Get confusion matrix
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=unique_categories)

    # Get classification report with zero_division=0 to avoid warnings
    report = classification_report(
        y_true_cat,
        y_pred_cat,
        labels=unique_categories,
        output_dict=True,
        zero_division=0
    )

    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_categories,
                yticklabels=unique_categories)
    plt.title('Confusion Matrix (Health Index Categories)')
    plt.xlabel('Predicted Category')
    plt.ylabel('True Category')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "confusion_matrix.png"))
    plt.close()

    # Visualize classification report
    plt.figure(figsize=(12, 6))
    categories = [cat for cat in unique_categories if cat in report]  # Only include categories in the report

    if len(categories) > 0:  # Only create the plot if there are categories to show
        # Plot precision, recall, and f1-score
        bar_width = 0.25
        index = np.arange(len(categories))

        precision_bars = plt.bar(index, [report[cat]['precision'] for cat in categories],
                                bar_width, label='Precision')
        recall_bars = plt.bar(index + bar_width, [report[cat]['recall'] for cat in categories],
                                bar_width, label='Recall')
        f1_bars = plt.bar(index + 2*bar_width, [report[cat]['f1-score'] for cat in categories],
                            bar_width, label='F1-score')

        plt.xlabel('Health Index Category')
        plt.ylabel('Score')
        plt.title('Classification Report by Category')
        plt.xticks(index + bar_width, categories)
        plt.legend()
        plt.ylim(0, 1.1)
        plt.grid(axis='y')
    else:
        plt.text(0.5, 0.5, "No categories available for metrics visualization",
                ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "classification_report.png"))
    plt.close()

    # Save metrics to CSV
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(output_path, "classification_report.csv"))

    # Return categorical metrics
    cat_metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }

    return cat_metrics


def plot_error_distribution(y_true, y_pred, output_path=None):
    """
    Visualize the distribution of prediction errors.

    Args:
        y_true (np.ndarray): True health index values.
        y_pred (np.ndarray): Predicted health index values.
        output_path (str): Path to save the visualizations.
    """
    # Set default output path if none provided
    if output_path is None:
        output_path = EVAL_PATH

    # Calculate errors
    errors = y_true - y_pred

    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, color='blue')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Error Distribution (Actual - Predicted)')
    plt.xlabel('Error Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "error_distribution.png"))
    plt.close()

    # Plot error vs predicted value
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, errors, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Prediction Error vs Predicted Value')
    plt.xlabel('Predicted Health Index')
    plt.ylabel('Error (Actual - Predicted)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "error_vs_predicted.png"))
    plt.close()

    # Calculate error percentiles
    percentiles = [5, 25, 50, 75, 95]
    error_percentiles = np.percentile(np.abs(errors), percentiles)

    # Create percentile plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        [f"{p}th" for p in percentiles],
        error_percentiles
    )

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.1,
            f"{height:.2f}",
            ha='center',
            va='bottom'
        )

    plt.title('Percentiles of Absolute Error')
    plt.ylabel('Absolute Error Value')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "error_percentiles.png"))
    plt.close()


def compare_with_baseline(y_true, y_pred, baseline_path=None, output_path=None):
    """
    Compare the CNN model with a baseline model.

    Args:
        y_true (np.ndarray): True health index values.
        y_pred (np.ndarray): Predicted health index values from CNN.
        baseline_path (str, optional): Path to baseline model predictions.
        output_path (str): Path to save the visualizations.
    """
    # Set default output path if none provided
    if output_path is None:
        output_path = EVAL_PATH

    # If no baseline provided, use mean predictor as baseline
    if baseline_path is None:
        baseline_pred = np.ones_like(y_true) * np.mean(y_true)
        baseline_name = "Mean Predictor"
    else:
        # Load baseline predictions
        baseline_df = pd.read_csv(baseline_path)
        baseline_pred = baseline_df['predicted'].values
        baseline_name = os.path.basename(baseline_path).split('.')[0]

    # Calculate metrics for both models
    cnn_mse = mean_squared_error(y_true, y_pred)
    cnn_rmse = np.sqrt(cnn_mse)
    cnn_mae = mean_absolute_error(y_true, y_pred)
    cnn_r2 = r2_score(y_true, y_pred)

    baseline_mse = mean_squared_error(y_true, baseline_pred)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_mae = mean_absolute_error(y_true, baseline_pred)
    baseline_r2 = r2_score(y_true, baseline_pred)

    # Create comparison bar chart
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(4)

    cnn_bars = plt.bar(index, [cnn_mse, cnn_rmse, cnn_mae, cnn_r2],
                        bar_width, label='CNN Model')
    baseline_bars = plt.bar(index + bar_width,
                             [baseline_mse, baseline_rmse, baseline_mae, baseline_r2],
                             bar_width, label=baseline_name)

    # Add value labels on top of each bar
    for bars in [cnn_bars, baseline_bars]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=9
            )

    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Model Comparison: CNN vs Baseline')
    plt.xticks(index + bar_width/2, ['MSE', 'RMSE', 'MAE', 'R²'])
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "model_comparison.png"))
    plt.close()

    # Create improvement percentage visualization
    improvement_mse = (baseline_mse - cnn_mse) / baseline_mse * 100
    improvement_rmse = (baseline_rmse - cnn_rmse) / baseline_rmse * 100
    improvement_mae = (baseline_mae - cnn_mae) / baseline_mae * 100

    # R² improvement needs different handling as it can be negative
    if baseline_r2 <= 0:
        improvement_r2 = 100  # Assume 100% improvement if baseline R² is negative or zero
    else:
        improvement_r2 = (cnn_r2 - baseline_r2) / abs(baseline_r2) * 100

    improvements = [improvement_mse, improvement_rmse, improvement_mae, improvement_r2]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        ['MSE', 'RMSE', 'MAE', 'R²'],
        improvements,
        color=['green' if imp > 0 else 'red' for imp in improvements]
    )

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.1 if height > 0 else height - 2,
            f"{height:.1f}%",
            ha='center',
            va='bottom' if height > 0 else 'top'
        )

    plt.title(f'Improvement Over {baseline_name} (%)')
    plt.ylabel('Improvement Percentage')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "improvement_percentage.png"))
    plt.close()


def main():
    """
    Main function to evaluate a trained CNN model.
    """
    # Define paths for test data and predictions
    test_data_path = os.path.join(MODEL_PATH, "test_data.csv")
    predictions_path = os.path.join(MODEL_PATH, "cnn_predictions.csv")

    # Check if the necessary files exist
    if not os.path.exists(test_data_path) or not os.path.exists(predictions_path):
        print("Error: Required files not found.")
        print(f"Missing one or both of these files:")
        print(f"  - {test_data_path}")
        print(f"  - {predictions_path}")
        print("\nPlease run the CNN model training first to generate these files.")
        print("You can do this by running: python src/models/cnn_model.py")
        return

    # Load the test data and predictions
    test_data = pd.read_csv(test_data_path)
    predictions = pd.read_csv(predictions_path)

    # Extract the target and predictions
    if 'actual' in predictions.columns and 'predicted' in predictions.columns:
        y_true = predictions['actual'].values
        y_pred = predictions['predicted'].values
    elif 'Health_index' in test_data.columns:
        y_true = test_data['Health_index'].values
        if 'predicted' in predictions.columns:
            y_pred = predictions['predicted'].values
        else:
            print("Error: Could not find predicted values in the predictions file.")
            return
    else:
        print("Error: Could not find actual health index values in the test data.")
        return

    # Evaluate the model
    print("Calculating regression metrics...")
    reg_metrics = evaluate_model_metrics(y_true, y_pred)

    print("Calculating categorical metrics...")
    cat_metrics = evaluate_categorical_metrics(y_true, y_pred)

    print("Plotting error distribution...")
    plot_error_distribution(y_true, y_pred)

    # Compare with baseline if available
    baseline_path = os.path.join(MODEL_PATH, "linear_regression_predictions.csv")
    if os.path.exists(baseline_path):
        print("Comparing with baseline model...")
        compare_with_baseline(y_true, y_pred, baseline_path)
    else:
        print("No baseline model found. Using mean predictor as baseline...")
        compare_with_baseline(y_true, y_pred)

    print("Evaluation complete!")
    print(f"Results saved to {EVAL_PATH}")


if __name__ == "__main__":
    main()