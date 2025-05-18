"""
Enhanced 1D Convolutional Neural Network for Power Transformer Health Index Prediction.

This module implements an enhanced 1D CNN architecture with batch normalization
and customized learning rate scheduling for better performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set random seed for reproducibility
np.random.seed(42)
keras.utils.set_random_seed(42)

# Define paths
OUTPUT_PATH = "../../outputs"
MODEL_PATH = os.path.join(OUTPUT_PATH, "models")
GRAPH_PATH = os.path.join(OUTPUT_PATH, "graphs", "cnn_model")

# Create directories if they don't exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)


def preprocess_data_for_cnn(df, target_column="Health_index", test_size=0.2):
    """
    Preprocess the data for CNN model by reshaping features.

    Args:
        df (pd.DataFrame): DataFrame containing the features and target.
        target_column (str): Name of the target column.
        test_size (float): Proportion of data to use for testing.

    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names
    """
    # Remove the target column to get features
    feature_names = df.drop(columns=[target_column]).columns
    X = df[feature_names].values
    y = df[target_column].values

    # Reshape for CNN input (samples, sequence_length, features)
    X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=test_size, random_state=42
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_names


def build_cnn_model(input_shape):
    """
    Build an enhanced 1D CNN model for regression with batch normalization.

    Args:
        input_shape (tuple): Shape of the input data (sequence_length, features).

    Returns:
        keras.Model: Compiled CNN model.
    """
    # Create model with Input layer first
    inputs = keras.layers.Input(shape=input_shape)

    # First convolutional layer
    x = keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    # Second convolutional layer
    x = keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    # Third convolutional layer
    x = keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    # Flatten the output and feed to dense layers
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(1)(x)  # Output layer for regression

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile model with Adam optimizer and MSE loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


def train_cnn_model(X_train, y_train, X_test, y_test, input_shape, epochs=100, batch_size=16):
    """
    Train the CNN model with early stopping and custom learning rate scheduling.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing target.
        input_shape (tuple): Shape of input data.
        epochs (int): Maximum number of epochs.
        batch_size (int): Batch size for training.

    Returns:
        tuple: Trained model and training history.
    """
    # Build model
    model = build_cnn_model(input_shape)

    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_PATH, "cnn_model_best.keras"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Custom learning rate scheduler
    def lr_schedule(epoch):
        initial_lr = 0.001
        if epoch < 10:
            return initial_lr
        elif epoch < 20:
            return initial_lr * 0.1
        else:
            return initial_lr * 0.01

    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, model_checkpoint, lr_scheduler],
        verbose=1
    )

    # Save the final model
    model.save(os.path.join(MODEL_PATH, "cnn_model_final.keras"))

    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.

    Args:
        model (tf.keras.Model): Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test target.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    # Make predictions
    y_pred = model.predict(X_test).flatten()

    # Save test data and predictions for later evaluation
    test_data = pd.DataFrame(X_test.reshape(X_test.shape[0], X_test.shape[1]))
    test_data['Health_index'] = y_test
    test_data.to_csv(os.path.join(MODEL_PATH, "test_data.csv"), index=False)

    predictions_df = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
    predictions_df.to_csv(os.path.join(MODEL_PATH, "cnn_predictions.csv"), index=False)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print metrics
    print(f"Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    return metrics, y_pred


def plot_training_history(history):
    """
    Plot the training and validation loss and metrics.

    Args:
        history (tf.keras.callbacks.History): Training history.
    """
    # Plot training & validation loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, "training_history.png"))
    plt.close()

    # Save the history data as CSV for future reference
    pd.DataFrame(history.history).to_csv(os.path.join(MODEL_PATH, "cnn_training_history.csv"), index=False)


def plot_predictions(y_test, y_pred):
    """
    Plot actual vs. predicted values.

    Args:
        y_test (np.ndarray): Actual values.
        y_pred (np.ndarray): Predicted values.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)

    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.title('Actual vs. Predicted Health Index')
    plt.xlabel('Actual Health Index')
    plt.ylabel('Predicted Health Index')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, "actual_vs_predicted.png"))
    plt.close()

    # Create residual plot
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Health Index')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, "residual_plot.png"))
    plt.close()


def save_model_summary(model, feature_names):
    """
    Save model summary and feature information.

    Args:
        model (tf.keras.Model): The trained model.
        feature_names (list): List of feature names.
    """
    # Save model summary
    with open(os.path.join(MODEL_PATH, "cnn_model_summary.txt"), 'w') as f:
        # Redirect the summary output to our file
        model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Add feature information
        f.write("\n\nFeatures used:\n")
        for i, feature in enumerate(feature_names):
            f.write(f"{i+1}. {feature}\n")


def predict_health_index(input_data, model_path=None):
    """
    Predict health index for new, unseen data.

    Args:
        input_data (pd.DataFrame): DataFrame containing features for prediction.
        model_path (str): Path to the saved model.

    Returns:
        np.ndarray: Predicted health index values.
    """
    # Set default model path if none provided
    if model_path is None:
        model_path = os.path.join(MODEL_PATH, "cnn_model_best.keras")

    # Load the model - using direct keras load_model function
    model = keras.saving.load_model(model_path)

    # Extract and reshape features
    X = input_data.values
    X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)

    # Make predictions
    predictions = model.predict(X_reshaped).flatten()

    return predictions


def main():
    """
    Main function to train and evaluate the CNN model.
    """
    print("Loading data...")
    # Load the preprocessed scaled data
    df = pd.read_csv(f"{MODEL_PATH}/scaled_data.csv")

    print("Preprocessing data for CNN...")
    X_train, X_test, y_train, y_test, feature_names = preprocess_data_for_cnn(df)

    print("Building and training CNN model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model, history = train_cnn_model(X_train, y_train, X_test, y_test, input_shape)

    print("Evaluating model...")
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    print("Generating plots...")
    plot_training_history(history)
    plot_predictions(y_test, y_pred)

    print("Saving model information...")
    save_model_summary(model, feature_names)

    print("Demonstrating prediction on sample data...")
    # Create sample data (using first 5 test samples as an example)
    sample_input = pd.DataFrame(
        X_test[:5, :, 0],  # Take first 5 samples, remove the channel dimension
        columns=feature_names
    )
    sample_predictions = predict_health_index(sample_input)

    print("\nSample predictions:")
    for i, pred in enumerate(sample_predictions):
        print(f"Sample {i+1}: Predicted Health Index = {pred:.2f}, Actual = {y_test[i]:.2f}")

    print("\nTraining and evaluation complete!")
    print(f"Model saved to {os.path.join(MODEL_PATH, 'cnn_model_best.keras')}")
    print(f"Plots saved to {GRAPH_PATH}")


if __name__ == "__main__":
    main()