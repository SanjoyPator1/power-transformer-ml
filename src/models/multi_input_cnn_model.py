"""
Multi-Input CNN for Power Transformer Health Index Prediction

This script implements a multi-input CNN model to predict the health index
of power transformers using different feature groups (gas, electrical, other).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, regularizers, Model, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Define paths
OUTPUT_PATH = "../../outputs"
MODEL_PATH = os.path.join(OUTPUT_PATH, "models")
GRAPH_PATH = os.path.join(OUTPUT_PATH, "graphs", "multi_input_cnn_model")

# Create directories if they don't exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)


def create_grouped_data(df, target_column="Health_index", test_size=0.2):
    """
    Create a grouped representation of the data for multi-input CNN.

    Args:
        df (pd.DataFrame): DataFrame containing the features and target.
        target_column (str): Name of the target column.
        test_size (float): Proportion of data to use for testing.

    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_groups, and scalers
    """
    # Define feature groups (domain-specific grouping)
    gas_features = ['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2',
                    'Ethylene', 'Ethane', 'Acethylene']
    electrical_features = ['Power_factor', 'Interfacial_V', 'Dielectric_rigidity']
    other_features = ['DBDS', 'Water_content', 'Life_expectation']

    # Create scalers for each feature group
    gas_scaler = StandardScaler()
    electrical_scaler = StandardScaler()
    other_scaler = StandardScaler()

    # Extract features
    X_gas = df[gas_features].values
    X_electrical = df[electrical_features].values
    X_other = df[other_features].values

    # Split the data first to avoid data leakage
    indices = np.arange(df.shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=SEED)

    # Scale each group separately to preserve domain-specific characteristics
    X_gas_train = gas_scaler.fit_transform(X_gas[train_indices])
    X_gas_test = gas_scaler.transform(X_gas[test_indices])

    X_el_train = electrical_scaler.fit_transform(X_electrical[train_indices])
    X_el_test = electrical_scaler.transform(X_electrical[test_indices])

    X_oth_train = other_scaler.fit_transform(X_other[train_indices])
    X_oth_test = other_scaler.transform(X_other[test_indices])

    # Get target values
    y_train = df[target_column].values[train_indices]
    y_test = df[target_column].values[test_indices]

    # Reshape the inputs correctly for 1D CNN (samples, features, 1)
    X_gas_train = X_gas_train.reshape(X_gas_train.shape[0], X_gas_train.shape[1], 1)
    X_gas_test = X_gas_test.reshape(X_gas_test.shape[0], X_gas_test.shape[1], 1)

    X_el_train = X_el_train.reshape(X_el_train.shape[0], X_el_train.shape[1], 1)
    X_el_test = X_el_test.reshape(X_el_test.shape[0], X_el_test.shape[1], 1)

    X_oth_train = X_oth_train.reshape(X_oth_train.shape[0], X_oth_train.shape[1], 1)
    X_oth_test = X_oth_test.reshape(X_oth_test.shape[0], X_oth_test.shape[1], 1)

    # CRITICAL FIX: Return train and test data as lists in the exact same order
    # as the model's input layers are defined
    X_train = {
        "gas_input": X_gas_train,
        "electrical_input": X_el_train,
        "other_input": X_oth_train
    }
    X_test = {
        "gas_input": X_gas_test,
        "electrical_input": X_el_test,
        "other_input": X_oth_test
    }

    feature_groups = {
        'gas': gas_features,
        'electrical': electrical_features,
        'other': other_features
    }

    scalers = {
        'gas': gas_scaler,
        'electrical': electrical_scaler,
        'other': other_scaler
    }

    return X_train, X_test, y_train, y_test, feature_groups, scalers


def multi_input_cnn(gas_shape, electrical_shape, other_shape):
    """
    Multi-input CNN model with inception-style blocks, squeeze-and-excitation, and residual connections for improved performance.

    Args:
        gas_shape (tuple): Shape of gas features input
        electrical_shape (tuple): Shape of electrical features input
        other_shape (tuple): Shape of other features input

    Returns:
        tf.keras.Model: Compiled multi-input CNN model
    """
    # --- Inception-style block for each branch ---
    def inception_block(x, filters, kernel_sizes, name_prefix):
        branches = []
        for i, k in enumerate(kernel_sizes):
            branch = layers.Conv1D(filters, k, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001), name=f'{name_prefix}_conv_{k}')(x)
            branch = layers.BatchNormalization(name=f'{name_prefix}_bn_{k}')(branch)
            branches.append(branch)
        concat = layers.Concatenate(name=f'{name_prefix}_concat')(branches)
        return concat

    # Squeeze-and-Excitation block
    def se_block(x, ratio=8, name_prefix=None):
        filters = x.shape[-1]
        se = layers.GlobalAveragePooling1D(name=f'{name_prefix}_se_gap')(x)
        se = layers.Dense(filters // ratio, activation='relu', name=f'{name_prefix}_se_dense1')(se)
        se = layers.Dense(filters, activation='sigmoid', name=f'{name_prefix}_se_dense2')(se)
        se = layers.Multiply(name=f'{name_prefix}_se_mult')([x, layers.Reshape((1, filters))(se)])
        return se

    # Gas branch
    gas_input = Input(shape=gas_shape, name="gas_input")
    x1 = inception_block(gas_input, 32, [1, 3, 5], 'gas')
    x1 = se_block(x1, name_prefix='gas')
    # Fix: set filters to 96 to match inception block output
    x1_res = layers.Conv1D(96, 1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001), name='gas_res_conv')(gas_input)
    x1_res = layers.BatchNormalization(name='gas_res_bn')(x1_res)
    x1 = layers.Add(name='gas_add')([x1, x1_res])
    x1 = layers.GlobalAveragePooling1D(name='gas_gap')(x1)
    x1 = layers.Dropout(0.2, name='gas_dropout')(x1)

    # Electrical branch
    electrical_input = Input(shape=electrical_shape, name="electrical_input")
    x2 = inception_block(electrical_input, 16, [1, 2], 'el')
    x2 = se_block(x2, name_prefix='el')
    x2_res = layers.Conv1D(32, 1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001), name='el_res_conv')(electrical_input)
    x2_res = layers.BatchNormalization(name='el_res_bn')(x2_res)
    x2 = layers.Add(name='el_add')([x2, x2_res])
    x2 = layers.GlobalAveragePooling1D(name='el_gap')(x2)
    x2 = layers.Dropout(0.2, name='el_dropout')(x2)

    # Other branch
    other_input = Input(shape=other_shape, name="other_input")
    x3 = inception_block(other_input, 16, [1, 2], 'oth')
    x3 = se_block(x3, name_prefix='oth')
    x3_res = layers.Conv1D(32, 1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001), name='oth_res_conv')(other_input)
    x3_res = layers.BatchNormalization(name='oth_res_bn')(x3_res)
    x3 = layers.Add(name='oth_add')([x3, x3_res])
    x3 = layers.GlobalAveragePooling1D(name='oth_gap')(x3)
    x3 = layers.Dropout(0.2, name='oth_dropout')(x3)

    # Concatenate branches
    combined = layers.concatenate([x1, x2, x3], name='combined_concat')

    # Dense block with increased capacity, LeakyReLU, dropout, L2 regularization, and residual connection
    x = layers.Dense(192, kernel_regularizer=regularizers.l2(0.0005), name='dense1')(combined)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.LeakyReLU(alpha=0.1, name='leakyrelu1')(x)
    x = layers.Dropout(0.15, name='dropout1')(x)
    dense1_out = x  # For residual

    x = layers.Dense(96, kernel_regularizer=regularizers.l2(0.0005), name='dense2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.LeakyReLU(alpha=0.1, name='leakyrelu2')(x)
    x = layers.Dropout(0.08, name='dropout2')(x)
    # Residual connection (if shapes match)
    if dense1_out.shape[-1] == x.shape[-1]:
        x = layers.Add(name='residual_add')([x, dense1_out])
    else:
        # Project dense1_out to match shape if needed
        proj = layers.Dense(96, kernel_regularizer=regularizers.l2(0.0005), name='residual_proj')(dense1_out)
        x = layers.Add(name='residual_add')([x, proj])

    x = layers.Dense(48, kernel_regularizer=regularizers.l2(0.0005), name='dense3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.LeakyReLU(alpha=0.1, name='leakyrelu3')(x)
    x = layers.Dropout(0.03, name='dropout3')(x)
    output = layers.Dense(1, name="health_index")(x)

    model = Model(inputs=[gas_input, electrical_input, other_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])

    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=16):
    """
    Train the model with callbacks.

    Args:
        model (tf.keras.Model): The model to train
        X_train (list): List of training inputs
        y_train (np.ndarray): Training targets
        X_test (list): List of validation inputs
        y_test (np.ndarray): Validation targets
        epochs (int): Maximum number of epochs
        batch_size (int): Batch size

    Returns:
        tuple: Trained model and training history
    """
    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and calculate performance metrics.

    Args:
        model (tf.keras.Model): Trained model
        X_test (list): Test input features
        y_test (np.ndarray): Test targets

    Returns:
        tuple: Metrics dictionary and predictions
    """
    # Make predictions
    y_pred = model.predict(X_test).flatten()

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print metrics
    print("\nMulti-Input CNN Model Evaluation:")
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


def plot_results(y_test, y_pred, history):
    """
    Create visualizations for model results.

    Args:
        y_test (np.ndarray): Actual values
        y_pred (np.ndarray): Predicted values
        history (tf.keras.callbacks.History): Training history
    """
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title('Actual vs Predicted Health Index')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(GRAPH_PATH, "actual_vs_predicted.png"))
    plt.close()

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train')
    plt.plot(history.history['val_mae'], label='Validation')
    plt.title('Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, "training_history.png"))
    plt.close()

    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(GRAPH_PATH, "residuals.png"))
    plt.close()


def predict_samples(model, sample_data, feature_groups, scalers):
    """
    Make predictions on sample data.

    Args:
        model (tf.keras.Model): Trained model
        sample_data (pd.DataFrame): Sample data to predict
        feature_groups (dict): Feature group definitions
        scalers (dict): Fitted feature scalers

    Returns:
        np.ndarray: Predicted values
    """
    # Extract and scale features
    gas_features = feature_groups['gas']
    electrical_features = feature_groups['electrical']
    other_features = feature_groups['other']

    X_gas = scalers['gas'].transform(sample_data[gas_features].values)
    X_electrical = scalers['electrical'].transform(sample_data[electrical_features].values)
    X_other = scalers['other'].transform(sample_data[other_features].values)

    # Reshape for CNN
    X_gas = X_gas.reshape(X_gas.shape[0], X_gas.shape[1], 1)
    X_electrical = X_electrical.reshape(X_electrical.shape[0], X_electrical.shape[1], 1)
    X_other = X_other.reshape(X_other.shape[0], X_other.shape[1], 1)

    # Make predictions
    input_dict = {
        "gas_input": X_gas,
        "electrical_input": X_electrical,
        "other_input": X_other
    }
    predictions = model.predict(input_dict).flatten()

    return predictions


def main():
    """Main function to execute the workflow."""
    print("\n" + "="*60)
    print("POWER TRANSFORMER HEALTH INDEX PREDICTION")
    print("="*60 + "\n")

    # Load data
    print("Loading data...")
    df = pd.read_csv(f"{MODEL_PATH}/scaled_data.csv")

    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, feature_groups, scalers = create_grouped_data(df)

    # Get input shapes
    gas_shape = X_train["gas_input"].shape[1:]
    electrical_shape = X_train["electrical_input"].shape[1:]
    other_shape = X_train["other_input"].shape[1:]

    print(f"Feature shapes:")
    print(f"Gas features: {gas_shape}")
    print(f"Electrical features: {electrical_shape}")
    print(f"Other features: {other_shape}")

    # Build and train model
    print("\nBuilding and training model...")
    model = multi_input_cnn(gas_shape, electrical_shape, other_shape)
    model.summary()

    model, history = train_model(model, X_train, y_train, X_test, y_test)

    # Evaluate model
    print("\nEvaluating model...")
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    # Plot results
    print("\nGenerating visualizations...")
    plot_results(y_test, y_pred, history)

    # Make sample predictions
    print("\nPredicting on sample data...")
    sample_indices = range(5)
    sample_data = df.iloc[sample_indices].copy()
    sample_predictions = predict_samples(model, sample_data, feature_groups, scalers)

    print("\nSample predictions:")
    for i, pred in enumerate(sample_predictions):
        actual = y_test[i] if i < len(y_test) else "Unknown"
        print(f"Sample {i + 1}: Predicted Health Index = {pred:.2f}, Actual = {actual}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()