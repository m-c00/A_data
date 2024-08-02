
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, Rbf, splrep, splev
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
import statsmodels.api as sm
from scipy.ndimage import uniform_filter1d

# List of file paths
file_paths = [
    
    'Rolling_avg_interpolated_data_c1.csv',
    'Rolling_avg_interpolated_data_c2.csv',
    'Rolling_avg_interpolated_data_c3.csv'
    # '03110027.csv'
    # '03110139_7_11_24_10_54.csv'
    # '03111605_7_12_24_1_0_am.csv'
    # '03110001_7_12_24_2_13.csv'
    # '03111312-7_12_24_4_28.csv'
    # '03111527_7_13_24_6_49_am.csv' # Missing comma added here
    # '03110017_7_14_24_9_14_am.csv' # Missing comma added here
    # '01020553_7_13_24_7_10_pm.csv'   # Missing comma added here
]



# Moving Average Filter
def moving_average(data, window_size):
    return uniform_filter1d(data, size=window_size)

# Iterate over each file and plot the data
for file_path in file_paths:
    data = pd.read_csv(file_path)   

    # Ensure all data is numeric and handle missing values
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.fillna(method='ffill')

    # Extract relevant columns (12, 13, 14) and rows (from 3 to the end)
    data_subset = data.iloc[1:, [0, 1]]

    # Plot columns with different colors, solid lines, and add legend
    plt.figure(figsize=(10, 6))
    plt.plot(data_subset.iloc[:, 0], label='IN LET', color='blue', linestyle='-')
    plt.plot(data_subset.iloc[:, 1], label='OUT LET', color='green', linestyle='-')
    # plt.plot(data_subset.iloc[:, 2], label='Water Bath', color='red', linestyle='-')
    # plt.plot(data_subset.iloc[:, 0] - data_subset.iloc[:, 1] + 25, label='Water Bath Offset', color='black', linestyle='-')

    plt.title(f'Configuration from {file_path}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()


    
    plt.figure(figsize=(10, 6))
    window_size = 30  # Adjust this to your needs
    x = np.arange(1, len(data_subset.iloc[:, 0]) + 1)


    moving_avg = moving_average(data_subset.iloc[:, 0], window_size)
    plt.plot(x, moving_avg, label=f'Moving Average (window size={window_size}) ')
    moving_avg = moving_average(data_subset.iloc[:, 1], window_size)
    plt.plot(x, moving_avg, label=f'Moving Average (window size={window_size}) ')
    plt.title(f'Configuration from {file_path}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    plt.grid(True)

    # Show the plot for each file
    plt.show()
