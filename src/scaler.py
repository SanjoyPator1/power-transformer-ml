from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_data(df):
    """Scales the data using StandardScaler."""
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return scaled_df, scaler
