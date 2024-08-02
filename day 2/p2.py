import pandas as pd
import matplotlib.pyplot as plt

def plot_sensor_data(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Ensure all data is numeric and handle missing values
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.fillna(method='ffill')
    
    # Extract relevant columns (4, 5, 6) and rows (from 3 to the end)
    data_subset = data.iloc[2:, [3, 4, 5]]
    
    # Plot columns with different colors, solid lines, and add legend
    plt.figure(figsize=(10, 6))
    plt.plot(data_subset.iloc[:, 0], label='IN LET', color='blue', linestyle='-')
    plt.plot(data_subset.iloc[:, 1], label='OUT LET', color='green', linestyle='-')
    
    plt.plot(data_subset.iloc[:, 2], label='Wather Bath', color='red', linestyle='-')
    plt.plot(data_subset.iloc[:, 0] - data_subset.iloc[:, 1] + 25, label='Wather Bath', color='black', linestyle='-')
    
    plt.title(f'Filtered Sensor Data from {file_path}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    

# List of file paths to plot
file_paths = ['c2p1.csv', 'GHE_DATA_C1_C3_EXP.csv', 'c2p2.csv' , 'recreated_setup2.csv']

# Plot data for each file
for file_path in file_paths:
    plot_sensor_data(file_path)


plt.show()