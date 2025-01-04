import pandas as pd

def check(df):
    """Provides a summary of the dataset."""
    columns = df.columns
    result = []
    for col in columns:
        dtypes = df[col].dtypes
        nunique = df[col].nunique()
        sum_null = df[col].isnull().sum()
        result.append([col, dtypes, nunique, sum_null])
    return pd.DataFrame(result, columns=['column', 'dtypes', 'nunique', 'sum_null'])
