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

# Compute rolling averages for the first subset
n = 30  # Adjust as needed
rolling_avg_inlet = compute_rolling_averages(data_subset1.iloc[:, 0], n)
rolling_avg_outlet = compute_rolling_averages(data_subset1.iloc[:, 1], n)

# Create a plot with dual x-axes for the first subset
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot original data
ax1.plot(data_subset1.index, data_subset1.iloc[:, 0], label='IN LET', color='blue', linestyle='-')
ax1.plot(data_subset1.index, data_subset1.iloc[:, 1], label='OUT LET', color='green', linestyle='-')
ax1.set_xlabel('Original Data Index')
ax1.set_ylabel('Value')
ax1.legend(loc='upper left')
ax1.grid(True)

# Create a second x-axis for the rolling averages
ax2 = ax1.twiny()
ax2.plot(rolling_avg_inlet.index * n, rolling_avg_inlet, label='IN LET (Rolling Avg)', color='red', linestyle='-')
ax2.plot(rolling_avg_outlet.index * n, rolling_avg_outlet, label='OUT LET (Rolling Avg)', color='black', linestyle='-')
ax2.set_xlabel('Rolling Average Index')
ax2.legend(loc='upper right')

plt.title('Configuration 1 with Dual X-Axis')
plt.show()

# Extract relevant columns (10, 11) and rows (from 3 to the end)
data_subset2 = data.iloc[2:, [10, 11]]

# Compute rolling averages for the second subset
rolling_avg_inlet2 = compute_rolling_averages(data_subset2.iloc[:, 0], n)
rolling_avg_outlet2 = compute_rolling_averages(data_subset2.iloc[:, 1], n)

# Create a plot with dual x-axes for the second subset
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot original data
ax1.plot(data_subset2.index, data_subset2.iloc[:, 0], label='IN LET', color='blue', linestyle='-')
ax1.plot(data_subset2.index, data_subset2.iloc[:, 1], label='OUT LET', color='green', linestyle='-')
ax1.set_xlabel('Original Data Index')
ax1.set_ylabel('Value')
ax1.legend(loc='upper left')
ax1.grid(True)

# Create a second x-axis for the rolling averages
ax2 = ax1.twiny()
ax2.plot(rolling_avg_inlet2.index * n, rolling_avg_inlet2, label='IN LET (Rolling Avg)', color='red', linestyle='-')
ax2.plot(rolling_avg_outlet2.index * n, rolling_avg_outlet2, label='OUT LET (Rolling Avg)', color='black', linestyle='-')
ax2.set_xlabel('Rolling Average Index')
ax2.legend(loc='upper right')

plt.title('Configuration 2 with Dual X-Axis')
plt.show()
