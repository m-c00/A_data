import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# Load the CSV file
file_path = 'GHE_DATA_C1_C3_EXP.csv'
data = pd.read_csv(file_path)

# Ensure all data is numeric and handle missing values
data = data.apply(pd.to_numeric, errors='coerce')
data = data.ffill()

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

# Define the data cleaning and interpolation function using PchipInterpolator
def clean_and_interpolate(column, start_idx, end_idx, num_extra_points, prev_points, next_points):
    """
    Cleans and interpolates the given column based on the specified indices and number of extra points.
    
    Parameters:
    column (pd.Series): The input column to clean and interpolate.
    start_idx (int): The starting index of the rows to replace or add data.
    end_idx (int): The ending index of the rows to replace or add data.
    num_extra_points (int): The number of extra data points to add.
    prev_points (int): The number of previous data points to consider for interpolation.
    next_points (int): The number of next data points to consider for interpolation.
    
    Returns:
    pd.Series: The cleaned and interpolated column.
    """
    # Ensure the input is a Pandas Series
    if not isinstance(column, pd.Series):
        column = pd.Series(column)
    
    # Ensure indices are within bounds
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, len(column) - 1)
    
    # Get the data points for interpolation
    interpolation_indices = list(range(start_idx - prev_points, start_idx)) + \
                            list(range(end_idx + 1, end_idx + 1 + next_points))
    interpolation_indices = [i for i in interpolation_indices if i >= 0 and i < len(column)]
    interpolation_values = column.iloc[interpolation_indices].values

    extra_indices = list(range(start_idx, end_idx + 1 + num_extra_points))

    interpolation_indices_ =list(range(start_idx - prev_points, start_idx)) + \
                             list(range(end_idx + 1 + num_extra_points, end_idx + num_extra_points+ next_points + 1))

    # Perform monotone interpolation
    interpolator = PchipInterpolator(interpolation_indices_, interpolation_values)

    # Interpolate values
    interpolated_values = interpolator(extra_indices)
    
    # Create a new series to accommodate the new data
    new_series = pd.Series(index=range(len(column) + len(extra_indices) - (end_idx - start_idx + 1)))
    new_series.iloc[:start_idx] = column.iloc[:start_idx]
    new_series.iloc[start_idx:start_idx + len(extra_indices)] = interpolated_values
    new_series.iloc[start_idx + len(extra_indices):] = column.iloc[end_idx + 1:]
    
    return new_series

# Extract relevant columns (4, 5, 6) and rows (from 3 to the end)
data_subset1 = data.iloc[2:, [3, 4, 5]].reset_index(drop=True)

# Plot raw data for the first subset
plt.figure(figsize=(10, 6))
plt.plot(data_subset1.index, data_subset1.iloc[:, 0], label='IN LET', color='blue', linestyle='-')
plt.plot(data_subset1.index, data_subset1.iloc[:, 1], label='OUT LET', color='green', linestyle='-')
plt.title('Configuration 1 - Raw Data')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)


# Interpolate and clean data for columns 0 and 1
interpolated_col_1 = clean_and_interpolate(data_subset1.iloc[:, 0], 5300, 5500, 1000, 500, 500)
interpolated_col_2 = clean_and_interpolate(data_subset1.iloc[:, 1], 5300, 5500, 1000, 500, 500)

# Concatenate original DataFrame with interpolated values
new_data_subset1 = pd.concat([interpolated_col_1, interpolated_col_2], axis=1)
new_data_subset1.columns = ['IN LET', 'OUT LET']

# Further interpolate and clean data for columns 0 and 1 in the concatenated DataFrame
new_data_subset1['IN LET'] = clean_and_interpolate(new_data_subset1['IN LET'], 13550, 13750, 900, 500, 500)
new_data_subset1['OUT LET'] = clean_and_interpolate(new_data_subset1['OUT LET'], 13550, 13750, 900, 500, 500)



# Compute rolling averages for the first subset
n = 30 # Adjust as needed
rolling_avg_inlet = compute_rolling_averages(new_data_subset1.iloc[:, 0], n)
rolling_avg_outlet = compute_rolling_averages(new_data_subset1.iloc[:, 1], n)

# Create a plot with dual x-axes for the first subset
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot original data
ax1.plot(range(len(data_subset1)), new_data_subset1.iloc[:len(data_subset1), 0], label='IN LET', color='blue', linestyle='-')
ax1.plot(range(len(data_subset1)), new_data_subset1.iloc[:len(data_subset1), 1], label='OUT LET', color='green', linestyle='-')
ax1.set_xlabel('Original Data Index')
ax1.set_ylabel('Value')
ax1.legend(loc='upper left')
ax1.grid(True)

# Create a second x-axis for the rolling averages
ax2 = ax1.twiny()
ax2.plot(rolling_avg_inlet.index * n, rolling_avg_inlet, label='IN LET (Rolling Avg)', color='red', linestyle='--')
ax2.plot(rolling_avg_outlet.index * n, rolling_avg_outlet, label='OUT LET (Rolling Avg)', color='black', linestyle='--')
ax2.set_xlabel('Rolling Average Index')
ax2.legend(loc='upper right')

plt.title('Configuration 1 with Dual X-Axis')
plt.show()
