import numpy as np
import pandas as pd

def create_1d_array():
    """
    Create a 1D NumPy array with values [1, 2, 3, 4, 5]
    Returns:
        numpy.ndarray: 1D array
    """
    return np.array([1,2,3,4,5])

def create_2d_array():
    """
    Create a 2D NumPy array with shape (3,3) of consecutive integers
    Returns:
        numpy.ndarray: 2D array
    """
    return np.arange(1,10).reshape(3,3)

def array_operations(arr):
    """
    Perform basic array operations:
    1. Calculate mean
    2. Calculate standard deviation
    3. Find max value
    Returns:
        tuple: (mean, std_dev, max_value)
    """
    mean = np.mean(arr)
    std_dev = np.std(arr)
    max_value = np.max(arr)
    return mean, std_dev, max_value

def read_csv_file(filepath):
    """
    Read a CSV file using Pandas
    Args:
        filepath (str): Path to CSV file
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    return pd.read_csv(filepath)

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame
    1. Identify number of missing values
    2. Fill missing values with appropriate method
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    missing_values = df.isnull().sum() # Identify number of missing values
    print("Missing values per column:\n", missing_values)

    df.fillna(df.mean(numeric_only=True), inplace=True) # Fill numeric columns with mean
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0]) # Fill categorical columns with mode
    return df

def select_data(df):
    """
    Select specific columns and rows from DataFrame
    Returns:
        pandas.DataFrame: Selected data
    """
    return df.iloc[:5, :2] # Select the first 5 rows and the first 2 columns from the dataframe


def rename_columns(df):
    """
    Rename columns of the DataFrame
    Returns:
        pandas.DataFrame: DataFrame with renamed columns
    """
    df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'), inplace=True) # Converts column names with lowercases, removes spaces, and replaces them with underscores
    return df
