import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, Rbf, BSpline, make_interp_spline
from numpy.polynomial.polynomial import Polynomial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as GPR_RBF


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



# Load the CSV file
file_path = 'clean_data_c1.csv'
data = pd.read_csv(file_path)

# Ensure all data is numeric and handle missing values
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(method='ffill')

data_subset1 = data.iloc[1:, [0, 1]]



interpolated_col_1_ = fit_and_interpolate(data_subset1.iloc[:, 0], 'rbf', 25920)
interpolated_col_2_ = fit_and_interpolate(data_subset1.iloc[:, 1], 'rbf', 25920)
# Save                    data to a CSV file
interpolated_df = pd.DataFrame({
    'IN_LET': interpolated_col_1_,
    'OUT_LET': interpolated_col_2_
})
interpolated_df.to_csv('interpolated_data_c1.csv', index=False)

print("interpolated data saved to 'interpolated_data.csv'")




# Create a plot with dual x-axes for the first subset
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(data_subset1.iloc[:, 0], label='IN LET ', color='green', linestyle='-')
ax1.plot(data_subset1.iloc[:, 1], label='OUT LET', color= 'green', linestyle='-')
ax1.set_xlabel('Clean Data Index')
ax1.set_ylabel('Value')
ax1.legend(loc='upper left')
ax1.grid(True)

ax2 = ax1.twiny()
x = np.arange(1, len(interpolated_col_2_) + 1)
ax2.scatter(x, interpolated_col_1_, color='black', label=f'Data ')
ax2.scatter(x, interpolated_col_2_, color='black', label=f'Data ')
ax2.set_xlabel('Interpolated Data Index')
ax2.legend(loc='upper right')
ax2.grid(True)
plt.show()