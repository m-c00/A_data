import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, Rbf, splrep, splev
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
import statsmodels.api as sm
from scipy.ndimage import uniform_filter1d

# Load the CSV file
file_path = 'rolling_average_data.csv'
data = pd.read_csv(file_path)

# Extract columns (excluding the first row)
y1 = data.iloc[1:, 0].values
y2 = data.iloc[1:, 1].values

# Generate x-values as an array of indices
x = np.arange(1, len(y1) + 1)

# Moving Average Filter
def moving_average(data, window_size):
    return uniform_filter1d(data, size=window_size)

# Function to perform curve fitting and plotting
def perform_curve_fitting(x, y, label_suffix):
    

    # Polynomial fitting
    try:
        poly_coefficients = np.polyfit(x, y, 105)
        poly_fit = np.poly1d(poly_coefficients)
    except np.linalg.LinAlgError as e:
        print(f"Polynomial fit failed for {label_suffix}: {e}")
        poly_fit = None

    # Linear regression
    try:
        lin_reg = LinearRegression()
        lin_reg.fit(x.reshape(-1, 1), y)
        lin_reg_fit = lin_reg.predict(x.reshape(-1, 1))
    except Exception as e:
        print(f"Linear regression fit failed for {label_suffix}: {e}")
        lin_reg_fit = None

    # Non-linear least squares fitting (Exponential)
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    try:
        initial_guess = (1, 0.01, y.min())  # Reasonable initial guess
        exp_params, _ = curve_fit(exp_func, x, y, p0=initial_guess, bounds=(0, [np.inf, np.inf, np.inf]))
    except (RuntimeError, ValueError) as e:
        print(f"Exponential fit failed for {label_suffix}: {e}")
        exp_params = None

    # Spline fitting
    try:
        if np.all(np.diff(x) > 0):  # Check if x is strictly increasing
            spline = UnivariateSpline(x, y)
        else:
            raise ValueError("x must be strictly increasing for Spline fit")
    except Exception as e:
        print(f"Spline fit failed for {label_suffix}: {e}")
        spline = None

    # Radial Basis Function (RBF) Interpolation
    try:
        rbf = Rbf(x, y)
    except Exception as e:
        print(f"RBF interpolation failed for {label_suffix}: {e}")
        rbf = None

    # B-Spline fitting
    try:
        tck = splrep(x, y)
    except Exception as e:
        print(f"B-Spline fit failed for {label_suffix}: {e}")
        tck = None

    # LOESS (Local Regression)
    try:
        lowess = sm.nonparametric.lowess
        loess_fit = lowess(y, x, frac=0.3)
    except Exception as e:
        print(f"LOESS fit failed for {label_suffix}: {e}")
        loess_fit = None

    # Gaussian Process Regression
    try:
        gpr = GaussianProcessRegressor()
        gpr.fit(x.reshape(-1, 1), y)
        gpr_fit = gpr.predict(x.reshape(-1, 1))
    except Exception as e:
        print(f"Gaussian Process Regression fit failed for {label_suffix}: {e}")
        gpr_fit = None

    # Plotting
    plt.scatter(x, y, color='black', label=f'Data {label_suffix}')

    x_fit = np.linspace(x.min(), x.max(), 1000)

    # Polynomial fit
    if poly_fit is not None:
        plt.plot(x_fit, poly_fit(x_fit), label=f'Polynomial Fit (degree 2) {label_suffix}')

    # Linear regression fit
    if lin_reg_fit is not None:
        plt.plot(x, lin_reg_fit, label=f'Linear Regression {label_suffix}')

    # Non-linear least squares fit (Exponential)
    if exp_params is not None:
        plt.plot(x_fit, exp_func(x_fit, *exp_params), label=f'Exponential Fit {label_suffix}')

    # Spline fit
    if spline is not None:
        plt.plot(x_fit, spline(x_fit), label=f'Spline Fit {label_suffix}')

    # RBF interpolation fit
    if rbf is not None:
        plt.plot(x_fit, rbf(x_fit), label=f'RBF Interpolation {label_suffix}')

    # B-Spline fit
    if tck is not None:
        y_bspline = splev(x_fit, tck)
        plt.plot(x_fit, y_bspline, label=f'B-Spline Fit {label_suffix}')

    # LOESS fit
    if loess_fit is not None:
        plt.plot(loess_fit[:, 0], loess_fit[:, 1], label=f'LOESS Fit {label_suffix}')

    # Gaussian Process Regression fit
    if gpr_fit is not None:
        plt.plot(x, gpr_fit, label=f'Gaussian Process Regression {label_suffix}')

    # Moving Average fit
    window_size = 5  # Adjust this to your needs
    moving_avg = moving_average(y, window_size)
    plt.plot(x, moving_avg, label=f'Moving Average (window size={window_size}) {label_suffix}')


    




plt.figure(figsize=(14, 10))
# Perform curve fitting and plotting for y1
perform_curve_fitting(x, y1, 'y1')

# Perform curve fitting and plotting for y2
perform_curve_fitting(x, y2, 'y2')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Curve Fitting Comparison ')
plt.grid(True)
plt.show()
