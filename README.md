# Power Transformer ML

## Power Transformer Health Index Analysis
This project involves analyzing and predicting the health index of power transformers using various machine learning models. The dataset used for this analysis includes key features like gas concentrations, dielectric rigidity, and health-related factors, which are essential in evaluating transformer health.

## Project Structure
```text
├── README.md
├── dataset
│   └── health-index.csv          # Raw dataset
├── main.py                       # Entry point of the project
├── outputs
│   ├── graphs
│   │   ├── boxplots             # Boxplot visualizations
│   │   ├── correlation_heatmap.png  # Heatmap of feature correlations
│   │   ├── countplots           # Countplot visualizations
│   │   ├── histograms           # Histogram visualizations
│   │   ├── kdeplots             # KDE plot visualizations
│   │   ├── pairplots            # Pairplot visualizations
│   │   └── scatterplots         # Scatterplot visualizations
│   └── models
│       ├── scaled_data.csv      # Preprocessed scaled data
│       └── scaler.pkl           # Scaler used for data normalization
├── requirements.txt             # Python dependencies
└── src
    ├── __pycache__              # Compiled Python files
    ├── data_loader.py           # Script to load the dataset
    ├── eda.py                   # Script for exploratory data analysis (EDA)
    ├── models
    │   └── linear_regression.py # Linear regression model script
    ├── preprocess.py            # Data preprocessing script
    ├── scaler.py                # Script for scaling data
    ├── train.py                 # Script to train models
    └── visualizations.py        # Script for generating visualizations
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SanjoyPator1/power-transformer-ml.git
   ```

2. Install required dependencies
   ```bash
   pip install -r requirements.txt	
   ```
## Dataset
The dataset used for this project is located in the dataset/health-index.csv file. It contains data on various gases, health index values, and transformer-related features. The dataset is pre-processed before being used in machine learning models.

# Features
- Acetylene
- CO2
- CO
- DBDS
- Dielectric rigidity
- Ethane
- Ethylene
- Hydrogen
- Interfacial Voltage
- Life Expectation
- Methane
- Nitrogen
- Oxygen
- Power Factor
- Water Content
- Health Index

## Steps to Run the Project

### 1. Preprocess the Data  
   The first step is to load and preprocess the data. This involves handling missing values, scaling features, and splitting the dataset into training and test sets. The data preprocessing is managed by the `preprocess.py` script.

   **Script to run:**  
   `preprocess.py`  
   *(Add your preprocessing code here)*

### 2. Perform Exploratory Data Analysis (EDA)  
   In this step, you'll explore the dataset by visualizing relationships between features using boxplots, histograms, scatterplots, and other relevant graphs. The results of the analysis will be saved in the `outputs/graphs` folder.

   **Script to run:**  
   `eda.py`  
   *(Add your EDA code here)*

### 3. Train the Model  
   This stage involves training various machine learning models, such as linear regression, to predict the health index. The trained models will be saved in the `outputs/models` folder as `.pkl` files for future use.

   **Script to run:**  
   `train.py`  
   *(Add your model training code here)*

### 4. Scale the Data  
   To ensure that all features are on a comparable scale, feature scaling is applied. This step is handled by the `scaler.py` script, which also saves the scaler as a `.pkl` file for later use in predictions.

   **Script to run:**  
   `scaler.py`  
   *(Add your scaler code here)*

### 5. Generate Visualizations  
   Visualizations such as boxplots, scatter plots, and pair plots are generated to better understand the data and the results of the analysis. These plots are saved in the `outputs/graphs` directory for further examination.

   **Script to run:**  
   `visualizations.py`  
   *(Add your visualizations code here)*

### 6. Run the Project  
   The `main.py` script ties everything together and runs the entire project. It sequentially calls the other scripts to preprocess the data, perform EDA, train models, scale features, and generate visualizations.

   **Script to run:**  
   `main.py`  
   *(Add your main execution code here)*

## Results
The results of your analysis, including generated graphs and trained models, will be available in the 
$outputs/graphs$ 
and 
$outputs/models$ 
folders.

## Contributing

Feel free to fork the repository and contribute by creating pull requests. If you encounter any issues or have suggestions for improvements, please create an issue on GitHub.

## License

This project is licensed under the MIT License.





