import pandas as pd
from src.data_loader import load_data
from src.eda import check
from src.visualizations import save_boxplots, correlation_heatmap, save_scatterplots, save_kdeplot, save_histograms, \
    save_pairplot, save_countplot
from src.scaler import scale_data

# Define paths
DATA_PATH = "../dataset/health-index.csv"
OUTPUT_PATH = "../outputs/"
GRAPH_PATH = f"{OUTPUT_PATH}/graphs/"
MODEL_PATH = f"{OUTPUT_PATH}/models/"

def drop_outliers(df):
    """
    Drop rows from the dataframe based on predefined conditions.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with outliers removed.
    """
    # Define individual conditions
    df1 = df.index[(df['Hydrogen'] > 15000)]
    df2 = df.index[(df['Oxigen'] > 40000)]
    df3 = df.index[(df['Nitrogen'] > 80000)]
    df4 = df.index[(df['Methane'] > 2000)]
    df5 = df.index[(df['CO'] > 1250)]
    df6 = df.index[(df['CO2'] > 15000)]
    df7 = df.index[(df['Ethylene'] > 7500)]
    df8 = df.index[(df['Ethane'] > 3000)]
    df9 = df.index[(df['Acethylene'] > 6000)]
    df10 = df.index[(df['Power_factor'] > 30)]
    df11 = df.index[(df['Water_content'] > 100)]
    df12 = df.index[(df['Health_index'] > 80)]

    # Combine all indices
    outlier_indices = df1.union(df2).union(df3).union(df4).union(df5)\
                         .union(df6).union(df7).union(df8).union(df9)\
                         .union(df10).union(df11).union(df12)

    # Drop rows
    print(f"Number of rows to drop: {len(outlier_indices)}")
    return df.drop(outlier_indices)

def preprocess_and_visualize():
    """
    Perform the complete preprocessing pipeline:
    - Drop outliers
    - Visualizations
    - Data Scaling
    """

    # Step 1: Load data
    df = load_data(DATA_PATH)

    # Step 2: EDA
    eda_summary = check(df)
    print(eda_summary)

    # Step 3: Preprocessing
    df = drop_outliers(df)

    # Step 4: Visualizations
    numerical_columns = ['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene',
                         'DBDS', 'Power_factor', 'Interfacial_V', 'Dielectric_rigidity', 'Water_content',
                         'Health_index', 'Life_expectation']

    # Boxplots
    save_boxplots(df, numerical_columns, save_path=f"{GRAPH_PATH}/boxplots")

    # Correlation Heatmap
    correlation_heatmap(df, save_path=GRAPH_PATH)

    # Scatterplots
    save_scatterplots(df, save_path=f"{GRAPH_PATH}/scatterplots")

    # KDE Plots
    save_kdeplot(df, 'Hydrogen', 'Methane', 'Health_index',
                 'Hydrogen & Methane VS Health index', 'Hydrogen', 'Methane',
                 save_path=f"{GRAPH_PATH}/kdeplots/hydrogen_methane_health_index.png")
    save_kdeplot(df, 'DBDS', 'Interfacial_V', 'Health_index',
                 'DBDS & Interfacial_V VS Health index', 'DBDS', 'Interfacial_V',
                 save_path=f"{GRAPH_PATH}/kdeplots/dbds_interfacial_health_index.png")

    # Histograms
    save_histograms(df, save_path=f"{GRAPH_PATH}/histograms/all_columns_histogram.png")

    # Pairplots
    save_pairplot(df,
                  x_vars=["Hydrogen", "Oxigen", "Methane", "CO", "CO2", "Ethylene", "Ethane", "Acethylene"],
                  y_vars=["Hydrogen", "Oxigen", "Methane", "CO", "CO2", "Ethylene", "Ethane", "Acethylene"],
                  save_path=f"{GRAPH_PATH}/pairplots/gas_features_pairplot.png")
    save_pairplot(df,
                  x_vars=["DBDS", "Power_factor", "Interfacial_V", "Dielectric_rigidity", "Water_content",
                          "Health_index", "Life_expectation"],
                  y_vars=["DBDS", "Power_factor", "Interfacial_V", "Dielectric_rigidity", "Water_content",
                          "Health_index", "Life_expectation"],
                  save_path=f"{GRAPH_PATH}/pairplots/other_features_pairplot.png")
    save_pairplot(df,
                  x_vars=["DBDS", "Power_factor", "Interfacial_V", "Dielectric_rigidity", "Water_content",
                          "Health_index", "Life_expectation"],
                  y_vars=["Hydrogen", "Oxigen", "Methane", "CO", "CO2", "Ethylene", "Ethane", "Acethylene"],
                  save_path=f"{GRAPH_PATH}/pairplots/mixed_features_pairplot.png")

    # Countplots
    save_countplot(df, 'Health_index', 'Health_index', 'Health_index', 'Count',
                   save_path=f"{GRAPH_PATH}/countplots/health_index_countplot.png")
    save_countplot(df, 'Life_expectation', 'Life_expectation', 'Life_expectation', 'Count',
                   save_path=f"{GRAPH_PATH}/countplots/life_expectation_countplot.png")
    save_countplot(df, 'Water_content', 'Water_content', 'Water_content', 'Count',
                   save_path=f"{GRAPH_PATH}/countplots/water_content_countplot.png")
    save_countplot(df, 'Dielectric_rigidity', 'Dielectric_rigidity', 'Dielectric_rigidity', 'Count',
                   save_path=f"{GRAPH_PATH}/countplots/dielectric_rigidity_countplot.png")
    save_countplot(df, 'Interfacial_V', 'Interfacial_V', 'Interfacial_V', 'Count',
                   save_path=f"{GRAPH_PATH}/countplots/interfacial_v_countplot.png")

    # Step 5: Scaling
    scaled_df, scaler = scale_data(df)

    # Save the scaled data and scaler
    scaled_df.to_csv(f"{MODEL_PATH}/scaled_data.csv", index=False)
    import joblib
    joblib.dump(scaler, f"{MODEL_PATH}/scaler.pkl")

    # Print the shape of the final training data
    print("Shape of the final training data:", scaled_df.shape)


if __name__ == "__main__":
    preprocess_and_visualize()

