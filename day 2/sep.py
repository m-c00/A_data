import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, Rbf, BSpline, make_interp_spline
from numpy.polynomial.polynomial import Polynomial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as GPR_RBF

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


# Load the CSV file
file_path = 'GHE_DATA_C1_C3_EXP.csv'
data = pd.read_csv(file_path)

# Ensure all data is numeric and handle missing values
data = data.apply(pd.to_numeric, errors='coerce')
data = data.ffill()

methods = [
    'pchip', 'rbf', 'bspline', 'polynomial', 'gaussian_process'
]

# Extract relevant columns (4, 5, 6) and rows (from 3 to the end)
data_subset1 = data.iloc[2:, [3, 4, 5]].reset_index(drop=True)

# Generate x-values as an array of indices
x = np.arange(1, len(data_subset1.iloc[:, 0]) + 1)
for method in methods:
    # Interpolate and clean data
    interpolated_col_1 = clean_and_interpolate(data_subset1.iloc[:, 0], 13550, 13750, 100, 500, 500, method)
    interpolated_col_1 = clean_and_interpolate(interpolated_col_1, 5300, 5500, 100, 500, 500, method)

    interpolated_col_2 = clean_and_interpolate(data_subset1.iloc[:, 1], 13550, 13750, 100, 500, 500, method)
    interpolated_col_2 = clean_and_interpolate(interpolated_col_2, 5300, 5500, 100, 500, 500, method)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, data_subset1.iloc[:, 0], color='black', label=f'Data ')
    plt.scatter(x, data_subset1.iloc[:, 1], color='black', label=f'Data ')
    # Plot original data
    plt.plot(range(len(interpolated_col_1)), interpolated_col_1, label='IN LET', color='blue', linestyle='-')
    plt.plot(range(len(interpolated_col_2)), interpolated_col_2, label='OUT LET', color='green', linestyle='-')
    plt.xlabel(' Index')
    plt.ylabel('Value')
    plt.legend(loc='upper left')
    plt.title(method)
    plt.grid(True)

plt.show()