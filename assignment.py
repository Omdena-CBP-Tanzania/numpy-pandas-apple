import numpy as np
import pandas as pd

def create_1d_array():
    """
    Create a 1D NumPy array with values [1, 2, 3, 4, 5]
    Returns:
        numpy.ndarray: 1D array
    """
    return np.array([1, 2, 3, 4, 5])
    pass
array = create_1d_array()
print(array)

def create_2d_array():
    """
    Create a 2D NumPy array with shape (3,3) of consecutive integers
    Returns:
        numpy.ndarray: 2D array
    """
    return np.arange(1, 10).reshape(3, 3)
    pass
array2 = create_2d_array()
print(array2)

def array_operations(arr):
    """
    Perform basic array operations:
    1. Calculate mean
    2. Calculate standard deviation
    3. Find max value
    Args:
        arr (numpy.ndarray): Input NumPy array
    Returns:
        tuple: (mean, std_dev, max_value)
    """
    pass
    mean_value = np.mean(arr)
    std_dev = np.std(arr)
    max_value = np.max(arr)
    
    return mean_value, std_dev, max_value
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = array_operations(arr)
print(result)  # Output: (Mean, Standard Deviation, Max Value)

def read_csv_file(filepath):
    """
    Read a CSV file using Pandas
    Args:
        filepath (str): Path to CSV file
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    return pd.read_csv(filepath)
    pass

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame
    1. Identify number of missing values
    2. Fill missing values with appropriate method
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    # Step 1: Identify the number of missing values per column
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Step 2: Fill missing values
    # Fill numerical columns with the mean of each column
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Fill categorical columns with the mode (most frequent value) of each column
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df
    pass

def select_data(df, columns=None, rows=None):
    """
    Select specific columns and rows from DataFrame
    Returns:
        pandas.DataFrame: Selected data
    """
    # Select columns if specified
    if columns is not None:
        df = df[columns]
    
    # Select rows if specified
    if rows is not None:
        df = df.iloc[rows]
    
    return df
    """
    Select specific columns and rows from DataFrame
    Returns:
        pandas.DataFrame: Selected data
    """
    # Select columns if specified
    if columns:
        df = df[columns]
    
    # Select rows if specified
    if rows:
        df = df.iloc[rows]
    
    return df
    pass

def rename_columns(df):
    """
    Rename columns of the DataFrame
    Returns:
        pandas.DataFrame: DataFrame with renamed columns
    """
     # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()
    # Rename columns: convert to lowercase and replace spaces with underscores
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df
    pass
