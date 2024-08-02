import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV files
setup2_start_file_path = 'c2p2.csv'
setup2_end_file_path = 'c2p1.csv'

# Load data
setup2_start_data = pd.read_csv(setup2_start_file_path)
setup2_end_data = pd.read_csv(setup2_end_file_path)

# Ensure all data is numeric and handle missing values
setup2_start_data = setup2_start_data.apply(pd.to_numeric, errors='coerce').fillna(method='ffill')
setup2_end_data = setup2_end_data.apply(pd.to_numeric, errors='coerce').fillna(method='ffill')

# Extract relevant columns (3, 4, 5) and rows (from 2 to the end)
setup2_start_data = setup2_start_data.iloc[2:, [2, 3, 4]].reset_index(drop=True)
setup2_end_data = setup2_end_data.iloc[2:, [2, 3, 4]].reset_index(drop=True)

# Determine the length of the middle part to be interpolated
total_length_needed = 25000
start_length = len(setup2_start_data)
end_length = len(setup2_end_data)
middle_length_needed = total_length_needed - start_length - end_length

# Linear interpolation to estimate average values for the middle section
interpolated_middle_data = pd.DataFrame(index=range(middle_length_needed), columns=setup2_start_data.columns)
for column in setup2_start_data.columns:
    interpolated_middle_data[column] = np.linspace(setup2_start_data.iloc[-1][column], setup2_end_data.iloc[0][column], middle_length_needed)

# Function to refine interpolated values based on deviation
def refine_interpolated_values(start_data, end_data, interpolated_data):
    refined_data = interpolated_data.copy()
    for column in start_data.columns:
        # Calculate the average deviation from start and end data
        start_deviation = start_data[column].diff().mean()
        end_deviation = end_data[column].diff().mean()
        average_deviation = (start_deviation + end_deviation) / 2

        # Adjust interpolated values using average deviation
        for i in range(1, len(interpolated_data)):
            refined_data[column].iloc[i] = refined_data[column].iloc[i-1] + average_deviation

    return refined_data

# Refine the interpolated middle data
refined_middle_data = refine_interpolated_values(setup2_start_data, setup2_end_data, interpolated_middle_data)

# Combine all parts
recreated_setup2_data = pd.concat([setup2_start_data, refined_middle_data, setup2_end_data], ignore_index=True)

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
