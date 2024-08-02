import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'GHE_DATA_C1_C3_EXP.csv'
data = pd.read_csv(file_path)

# Ensure all data is numeric and handle missing values
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(method='ffill')

# Define the rolling average function
def compute_rolling_averages(column, n):
    """
    Computes the rolling averages for the given column based on the specified number of elements.
    
    Parameters:
    column (list or pd.Series): The input column to compute averages on.
    n (int): The number of elements to compute each average.
    
    Returns:
    pd.Series: A Pandas Series containing the rolling averages.
    """
    # Ensure the input is a Pandas Series
    if not isinstance(column, pd.Series):
        column = pd.Series(column)
    
    # Calculate the rolling averages
    averages = [column[i:i + n].mean() for i in range(0, len(column), n)]
    
    return pd.Series(averages)

# Extract relevant columns (4, 5, 6) and rows (from 3 to the end)
data_subset1 = data.iloc[2:, [3, 4, 5]]

# Plot raw data for the first subset
plt.figure(figsize=(10, 6))
plt.plot(data_subset1.iloc[:, 0], label='IN LET', color='blue', linestyle='-')
plt.plot(data_subset1.iloc[:, 1], label='OUT LET', color='green', linestyle='-')

# Compute and plot rolling averages for the first subset
n = 10  # Adjust as needed
rolling_avg_inlet = compute_rolling_averages(data_subset1.iloc[:, 0], n)
rolling_avg_outlet = compute_rolling_averages(data_subset1.iloc[:, 1], n)
plt.figure(figsize=(10, 6))
plt.plot(rolling_avg_inlet, label='IN LET (Rolling Avg)', color='blue', linestyle='--')
plt.plot(rolling_avg_outlet, label='OUT LET (Rolling Avg)', color='green', linestyle='--')

plt.title('Configuration 1')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Extract relevant columns (10, 11) and rows (from 3 to the end)
data_subset2 = data.iloc[2:, [10, 11]]

# Plot raw data for the second subset
plt.figure(figsize=(10, 6))
plt.plot(data_subset2.iloc[:, 0], label='IN LET', color='blue', linestyle='-')
plt.plot(data_subset2.iloc[:, 1], label='OUT LET', color='green', linestyle='-')

# Compute and plot rolling averages for the second subset
rolling_avg_inlet2 = compute_rolling_averages(data_subset2.iloc[:, 0], n)
rolling_avg_outlet2 = compute_rolling_averages(data_subset2.iloc[:, 1], n)
plt.figure(figsize=(10, 6))
plt.plot(rolling_avg_inlet2, label='IN LET (Rolling Avg)', color='blue', linestyle='--')
plt.plot(rolling_avg_outlet2, label='OUT LET (Rolling Avg)', color='green', linestyle='--')

plt.title('Configuration 2')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.show()
