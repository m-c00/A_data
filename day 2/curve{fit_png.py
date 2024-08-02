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
def perform_curve_fitting():
    # plt.figure(figsize=(10, 6))

    # Polynomial fitting
    def plot_polynomial_fit(x, y):
        try:
            poly_coefficients = np.polyfit(x, y, 105)
            poly_fit = np.poly1d(poly_coefficients)
        except np.linalg.LinAlgError as e:
            print(f"Polynomial fit failed for : {e}")
            return
        

        plt.plot(x, poly_fit(x), label=f'Polynomial Fit ')
        

    # Linear regression
    def plot_linear_regression(x, y):
        try:
            lin_reg = LinearRegression()
            lin_reg.fit(x.reshape(-1, 1), y)
            lin_reg_fit = lin_reg.predict(x.reshape(-1, 1))
        except Exception as e:
            print(f"Linear regression fit failed for : {e}")
            return

        plt.plot(x, lin_reg_fit, label=f'Linear Regression Fit ')
        
        
        

    # Non-linear least squares fitting (Exponential)
    def plot_exponential_fit(x, y):
        def exp_func(x, a, b, c):
            return a * np.exp(b * x) + c

        try:
            initial_guess = (1, 0.01, y.min())  # Reasonable initial guess
            exp_params, _ = curve_fit(exp_func, x, y, p0=initial_guess, bounds=(0, [np.inf, np.inf, np.inf]))
        except (RuntimeError, ValueError) as e:
            print(f"Exponential fit failed for : {e}")
            return

        plt.plot(x, exp_func(x, *exp_params), label=f'Exponential Fit ')
       
    # Spline fitting
    def plot_spline_fit(x, y):
        try:
            if np.all(np.diff(x) > 0):  # Check if x is strictly increasing
                spline = UnivariateSpline(x, y)
            else:
                raise ValueError("x must be strictly increasing for Spline fit")
        except Exception as e:
            print(f"Spline fit failed for: {e}")
            return

        plt.plot(x, spline(x), label=f'Spline Fit')


    # Radial Basis Function (RBF) Interpolation
    def plot_rbf_interpolation(x, y):
        try:
            rbf = Rbf(x, y)
        except Exception as e:
            print(f"RBF interpolation failed for : {e}")
            return
        plt.plot(x, rbf(x), label=f'RBF Interpolation ')

        
    
    # B-Spline fitting
    def plot_bspline_fit(x, y):
        try:
            tck = splrep(x, y)
        except Exception as e:
            print(f"B-Spline fit failed for : {e}")
            return

        
        y_bspline = splev(x, tck)
        plt.plot(x, y_bspline, label=f'B-Spline Fit')
      

    # LOESS (Local Regression)
    def plot_loess_fit(x, y):
        try:
            lowess = sm.nonparametric.lowess
            loess_fit = lowess(y, x, frac=0.3)
        except Exception as e:
            print(f"LOESS fit failed for : {e}")
            return
        
        plt.plot(loess_fit[:, 0], loess_fit[:, 1], label=f'LOESS Fit')
        
        

    # Gaussian Process Regression
    def plot_gaussian_process_regression(x, y):
        try:
            gpr = GaussianProcessRegressor()
            gpr.fit(x.reshape(-1, 1), y)
            gpr_fit = gpr.predict(x.reshape(-1, 1))
        except Exception as e:
            print(f"Gaussian Process Regression fit failed for : {e}")
            return
        
        plt.plot(x, gpr_fit, label=f'Gaussian Process Regression ')

        

    # # Add more fitting methods as needed

    # Plot each fitting method in a separate figure
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y1, color='black', label=f'Data ')
    plt.scatter(x, y2, color='black', label=f'Data ')
    plot_polynomial_fit(x, y1)
    plot_polynomial_fit(x, y2)
    plt.legend()
    plt.title(f'Polynomial Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    # plt.scatter(x, y1, color='black', label=f'Data ')
    # plt.scatter(x, y2, color='black', label=f'Data ')
    plot_linear_regression(x, y1)
    plot_linear_regression(x, y2)
    plt.legend()
    plt.title(f'Linear Regression Fit ')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.grid(True)
                           
    plt.figure(figsize=(10, 6))
    # plt.scatter(x, y1, color='black', label=f'Data ')
    # plt.scatter(x, y2, color='black', label=f'Data ')
    plot_exponential_fit(x, y1)
    plot_exponential_fit(x, y2)
    plt.legend()
    plt.title(f'Exponential Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)


    plt.figure(figsize=(10, 6))
    plt.scatter(x, y1, color='black', label=f'Data ')
    plt.scatter(x, y2, color='black', label=f'Data ')
    plot_spline_fit(x, y1)
    plot_spline_fit(x, y2)
    plt.legend()
    plt.title(f'Spline Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)


    plt.figure(figsize=(10, 6))
    plot_rbf_interpolation(x, y1)
    plot_rbf_interpolation(x, y2)
    plt.scatter(x, y1, color='black', label=f'Data ')
    plt.scatter(x, y2, color='black', label=f'Data ')
    plt.legend()
    plt.title(f'RBF Interpolation ')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)



    plt.figure(figsize=(10, 6))
    plt.scatter(x, y1, color='black', label=f'Data ')
    plt.scatter(x, y2, color='black', label=f'Data ')

    plot_bspline_fit(x, y1)
    plot_bspline_fit(x, y2)

    plt.legend()
    plt.title(f'B-Spline Fit ')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)


    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y1, color='black', label=f'Data ')
    plt.scatter(x, y2, color='black', label=f'Data ')
    plot_loess_fit(x, y1)
    plot_loess_fit(x, y2)
    plt.legend()
    plt.title(f'LOESS Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y1, color='black', label=f'Data ')
    plt.scatter(x, y2, color='black', label=f'Data ')
    plot_gaussian_process_regression(x, y1)
    plot_gaussian_process_regression(x, y2)
    plt.legend()
    plt.title(f'Gaussian Process Regression Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)


    plt.figure(figsize=(10, 6))
    window_size = 30  # Adjust this to your needs
    moving_avg = moving_average(y1, window_size)
    plt.plot(x, moving_avg, label=f'Moving Average (window size={window_size}) ')
    moving_avg = moving_average(y2, window_size)
    plt.plot(x, moving_avg, label=f'Moving Average (window size={window_size}) ')
    plt.title(f'Moving Average Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

# Perform curve fitting and plotting for y1
perform_curve_fitting()
