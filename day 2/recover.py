import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV files
setup1_file_path = 'GHE_DATA_C1_C3_EXP.csv'
setup2_start_file_path = 'c2p2.csv'
setup2_end_file_path = 'c2p1.csv'

# Load data
setup1_data = pd.read_csv(setup1_file_path)
setup2_start_data = pd.read_csv(setup2_start_file_path)
setup2_end_data = pd.read_csv(setup2_end_file_path)

# Ensure all data is numeric and handle missing values
setup1_data = setup1_data.apply(pd.to_numeric, errors='coerce').fillna(method='ffill')
setup2_start_data = setup2_start_data.apply(pd.to_numeric, errors='coerce').fillna(method='ffill')
setup2_end_data = setup2_end_data.apply(pd.to_numeric, errors='coerce').fillna(method='ffill')

# Extract relevant columns (3, 4, 5) from all datasets
setup1_data = setup1_data.iloc[2:, [2, 3, 4]].reset_index(drop=True)
setup2_start_data = setup2_start_data.iloc[2:, [2, 3, 4]].reset_index(drop=True)
setup2_end_data = setup2_end_data.iloc[2:, [2, 3, 4]].reset_index(drop=True)

# Determine the length of the middle part to be interpolated
total_length_needed = 25000
start_length = len(setup2_start_data)
end_length = len(setup2_end_data)
middle_length_needed = total_length_needed - start_length - end_length

# Use interpolation to create the middle part
# Assume the middle part is similar to the middle part of setup1
setup1_middle_data = setup1_data.iloc[start_length:start_length + middle_length_needed]

# Create the middle part by scaling the setup1 middle data to match the start and end of setup2
scaling_factor_start = setup2_start_data.mean() / setup1_data.iloc[:start_length].mean()
scaling_factor_end = setup2_end_data.mean() / setup1_data.iloc[-end_length:].mean()
scaling_factor = (scaling_factor_start + scaling_factor_end) / 2

# Apply the scaling factor
interpolated_middle_data = setup1_middle_data * scaling_factor

# Combine all parts
recreated_setup2_data = pd.concat([setup2_start_data, interpolated_middle_data, setup2_end_data], ignore_index=True)

# Save the recreated data to a new CSV file
recreated_setup2_data.to_csv('recreated_setup2.csv', index=False)

# Plot and verify the recreated data
plt.figure(figsize=(10, 6))
plt.plot(recreated_setup2_data.index, recreated_setup2_data.iloc[:, 0], label='Column 3')
plt.plot(recreated_setup2_data.index, recreated_setup2_data.iloc[:, 1], label='Column 4')
plt.plot(recreated_setup2_data.index, recreated_setup2_data.iloc[:, 2], label='Column 5')

plt.title('Recreated Setup 2 Data')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
