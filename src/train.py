# train.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Function to load scaled data and split into train/test
def load_and_split_data(model_path):
    # Load the scaled data
    scaled_df = pd.read_csv(f"{model_path}/scaled_data.csv")

    # Load the scaler (if needed)
    scaler = joblib.load(f"{model_path}/scaler.pkl")

    # Assume the target variable is 'Health_index'
    x = scaled_df.drop('Health_index', axis=1)
    y = scaled_df['Health_index'].values.reshape(-1, 1)

    # Split into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    print("Shape of training data:", x_train.shape)
    print("Shape of test data:", x_test.shape)

    return x_train, x_test, y_train, y_test, scaler
