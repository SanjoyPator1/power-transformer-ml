import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_boxplots(df, columns, save_path):
    """Saves boxplots of specified columns."""
    os.makedirs(save_path, exist_ok=True)
    i = 0
    while i < len(columns):
        fig, ax = plt.subplots(figsize=[15, 4])
        sns.boxplot(x=columns[i], data=df, color='cornflowerblue', ax=ax)
        plt.savefig(f"{save_path}/{columns[i]}_boxplot.png")
        plt.close()
        i += 1

def correlation_heatmap(df, save_path):
    """
    Generate and save a correlation heatmap for the given dataframe.

    Args:
        df (pd.DataFrame): Dataframe to calculate correlations from.
        save_path (str): Directory path to save the heatmap image.
    """
    plt.figure(figsize=(12, 10))
    corr = df.corr()  # Calculate the correlation matrix
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/correlation_heatmap.png")
    plt.close()

def save_scatterplots(df, save_path):
    """
    Generate and save scatterplots for all numerical column pairs in the dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        save_path (str): Directory path to save scatterplot images.
    """
    os.makedirs(save_path, exist_ok=True)
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for i, col_x in enumerate(numerical_columns):
        for col_y in numerical_columns[i + 1:]:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=col_x, y=col_y, data=df)
            plt.title(f"Scatterplot: {col_x} vs {col_y}")
            plt.xlabel(col_x)
            plt.ylabel(col_y)
            plt.savefig(f"{save_path}/scatter_{col_x}_vs_{col_y}.png")
            plt.close()

def save_kdeplot(df, x, y, hue, title, xlabel, ylabel, save_path):
    """
    Save a KDE plot with the given parameters.

    Args:
        df (pd.DataFrame): Input dataframe.
        x, y, hue (str): Columns for KDE plot axes and hue.
        title (str): Title of the plot.
        xlabel, ylabel (str): Labels for X and Y axes.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(15, 10), dpi=90)
    sns.kdeplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        shade=True,
        fill=True,
        common_norm=False,
        palette='crest',
        alpha=0.5,
        linewidth=0,
        warn_singular=False,
        linewidths=2
    )
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.legend([hue], loc='best')
    plt.title(title, backgroundcolor='blue', c='white', fontsize=20)
    plt.grid()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def save_histograms(df, save_path):
    """
    Save histograms for all columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        save_path (str): Path to save the plot.
    """
    df.hist(bins=30, figsize=(20, 15))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def save_pairplot(df, x_vars, y_vars, save_path):
    """
    Save a pairplot with the given variables.

    Args:
        df (pd.DataFrame): Input dataframe.
        x_vars, y_vars (list): Variables for pairplot axes.
        save_path (str): Path to save the plot.
    """
    sns.pairplot(df, x_vars=x_vars, y_vars=y_vars)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def save_countplot(df, column, title, xlabel, ylabel, save_path):
    """
    Save a count plot for the given column.

    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column for count plot.
        title (str): Title of the plot.
        xlabel, ylabel (str): Labels for X and Y axes.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(15, 10), dpi=90)
    ax = sns.countplot(x=column, data=df)
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.legend([column], loc='best')
    plt.title(title, backgroundcolor='blue', c='white', fontsize=20)
    plt.grid()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()