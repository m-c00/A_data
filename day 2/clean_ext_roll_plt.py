import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, Rbf, BSpline, make_interp_spline
from numpy.polynomial.polynomial import Polynomial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as GPR_RBF

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




def fit_and_interpolate(column, method='pchip', desired_length=None):
    """
    Cleans and interpolates the given column based on the specified method and desired length.

    Parameters:
    column (pd.Series): The input column to clean and interpolate.
    method (str): The interpolation method to use ('pchip', 'rbf', 'bspline', 'polynomial', 'gaussian_process').
    desired_length (int): The desired length of the output column.

    Returns:
    pd.Series: The cleaned and interpolated column with the specified length.
    """
    # Ensure the input is a Pandas Series
    if not isinstance(column, pd.Series):
        column = pd.Series(column)
    
    # Get the current length of the column
    current_length = len(column)
    
    # If no desired length is provided, return the original column
    if desired_length is None or desired_length <= current_length:
        return column

    # Create the indices for interpolation
    original_indices = np.arange(current_length)
    target_indices = np.linspace(0, current_length - 1, desired_length)

    # Perform interpolation based on the chosen method
    if method == 'pchip':
        interpolator = PchipInterpolator(original_indices, column)
    elif method == 'rbf':
        interpolator = Rbf(original_indices, column, function='multiquadric')
    elif method == 'bspline':
        t, c, k = make_interp_spline(original_indices, column, k=3).tck
        interpolator = BSpline(t, c, k)
    elif method == 'polynomial':
        coefs = Polynomial.fit(original_indices, column, deg=min(105, len(original_indices) - 1)).convert().coef
        interpolator = Polynomial(coefs)
    elif method == 'gaussian_process':
        kernel = GPR_RBF()
        gp = GaussianProcessRegressor(kernel=kernel)
        gp.fit(original_indices.reshape(-1, 1), column)
        interpolator = lambda x: gp.predict(x.reshape(-1, 1))
    else:
        raise ValueError("Invalid interpolation method. Choose from 'pchip', 'rbf', 'bspline', 'polynomial', 'gaussian_process'.")

    # Interpolate values
    interpolated_values = interpolator(target_indices)

    return pd.Series(interpolated_values, index=range(desired_length))


def clean_and_interpolate(column, start_idx, end_idx, num_extra_points, prev_points, next_points, method='pchip'):
    """
    Cleans and interpolates the given column based on the specified indices and number of extra points.

    Parameters:
    column (pd.Series): The input column to clean and interpolate.
    start_idx (int): The starting index of the rows to replace or add data.
    end_idx (int): The ending index of the rows to replace or add data.
    num_extra_points (int): The number of extra data points to add.
    prev_points (int): The number of previous data points to consider for interpolation.
    next_points (int): The number of next data points to consider for interpolation.
    method (str): The interpolation method to use ('pchip', 'rbf', 'bspline', 'polynomial', 'gaussian_process').

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

    interpolation_indices_ = list(range(start_idx - prev_points, start_idx)) + \
                             list(range(end_idx + 1 + num_extra_points, end_idx + num_extra_points + next_points + 1))

    # Perform interpolation based on the chosen method
    if method == 'pchip':
        interpolator = PchipInterpolator(interpolation_indices_, interpolation_values)
    elif method == 'rbf':
        interpolator = Rbf(interpolation_indices_, interpolation_values, function='multiquadric')
    elif method == 'bspline':
        t, c, k = make_interp_spline(interpolation_indices_, interpolation_values, k=3).tck
        interpolator = BSpline(t, c, k)
    elif method == 'polynomial':
        coefs = Polynomial.fit(interpolation_indices_, interpolation_values, deg=min(105, len(interpolation_indices_) - 1)).convert().coef
        interpolator = Polynomial(coefs)
    elif method == 'gaussian_process':
        kernel = GPR_RBF()
        gp = GaussianProcessRegressor(kernel=kernel)
        gp.fit(np.array(interpolation_indices_).reshape(-1, 1), interpolation_values)
        interpolator = lambda x: gp.predict(np.array(x).reshape(-1, 1))
    else:
        raise ValueError("Invalid interpolation method. Choose from 'pchip', 'rbf', 'bspline', 'polynomial', 'gaussian_process'.")

    # Interpolate values
    interpolated_values = interpolator(np.array(extra_indices))

    # Create a new series to accommodate the new data
    new_series = pd.Series(index=range(len(column) + len(extra_indices) - (end_idx - start_idx + 1)))
    new_series.iloc[:start_idx] = column.iloc[:start_idx]
    new_series.iloc[start_idx:start_idx + len(extra_indices)] = interpolated_values
    new_series.iloc[start_idx + len(extra_indices):] = column.iloc[end_idx + 1:]

    return new_series


# Extract relevant columns (4, 5, 6) and rows (from 3 to the end)
data_subset1 = data.iloc[2:, [3, 4, 5]]




# Interpolate and clean data
interpolated_col_1 = clean_and_interpolate(data_subset1.iloc[:, 0], 13550, 13750, 100, 500, 500, 'pchip')
interpolated_col_1 = clean_and_interpolate(interpolated_col_1, 5300, 5500, 100, 500, 500, 'pchip')

interpolated_col_2 = clean_and_interpolate(data_subset1.iloc[:, 1], 13550, 13750, 100, 500, 500,'pchip')
interpolated_col_2 = clean_and_interpolate(interpolated_col_2, 5300, 5500, 100, 500, 500, 'pchip')



interpolated_col_1_ = fit_and_interpolate(interpolated_col_1, 'rbf', 25920)
interpolated_col_2_ = fit_and_interpolate(interpolated_col_2, 'rbf', 25920)
# Save the rolling average data to a CSV file
interpolated_df = pd.DataFrame({
    'IN_LET': interpolated_col_1_,
    'OUT_LET': interpolated_col_1_
})
interpolated_df.to_csv('interpolated_data_c1.csv', index=False)

print("interpolated data saved to 'interpolated_data.csv'")


# Compute rolling averages for the first subset
n = 30 # Adjust as needed
rolling_avg_inlet = compute_rolling_averages(interpolated_col_1_, n)
rolling_avg_outlet = compute_rolling_averages(interpolated_col_2_, n)
# Save the rolling average data to a CSV file
rolling_avg_df = pd.DataFrame({
    'Rolling_Avg_IN_LET': rolling_avg_inlet,
    'Rolling_Avg_OUT_LET': rolling_avg_outlet
})
rolling_avg_df.to_csv('rolling_average_data_C1.csv', index=False)

print("Rolling average data saved to 'rolling_average_data.csv'")


# Create a plot with dual x-axes for the first subset
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(interpolated_col_1_, label='IN LET (gaussian_process)', color='green', linestyle='-')
ax1.plot(interpolated_col_2_, label='OUT LET (gaussian_process)', color= 'green', linestyle='-')
ax1.set_xlabel('Original Rolling Avg Index')
ax1.set_ylabel('Value')
ax1.legend(loc='upper left')
ax1.grid(True)

ax2 = ax1.twiny()
x = np.arange(1, len(interpolated_col_2) + 1)
ax2.scatter(x, interpolated_col_1, color='black', label=f'Data ')
ax2.scatter(x, interpolated_col_2, color='black', label=f'Data ')
ax2.set_xlabel('Rolling Average Index')
ax2.legend(loc='upper right')

ax3 = ax1.twiny() 

ax3.plot(rolling_avg_inlet, label='IN LET (gaussian_process)', color='red', linestyle='--')
ax3.plot(rolling_avg_outlet, label='OUT LET (gaussian_process)', color= 'red', linestyle='--')

plt.title('Rolling Average Comparison with Different Interpolation Methods')
plt.grid(True)
plt.show()