import pandas as pd
import matplotlib.pyplot as plt

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



# Load the CSV file
file_path = 'interpolated_data_c1.csv'
data = pd.read_csv(file_path)

# Ensure all data is numeric and handle missing values
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(method='ffill')

data_subset1 = data.iloc[1:, [0, 1]]
# Round the DataFrame to 1 decimal place
data_subset1_rounded = data_subset1.round(1)

# Save the rolling average data to a CSV file
Round_df = pd.DataFrame({
    'IN_LET': data_subset1_rounded.iloc[:, 0],
    'OUT_LET': data_subset1_rounded.iloc[:, 1]
})
Round_df.to_csv('Round_interpolated_data_c1.csv', index=False)

print("interpolated data saved to 'Round_interpolated_data.csv'")




# Compute rolling averages for the first subset
n = 30  # Adjust as needed
rolling_avg_inlet = compute_rolling_averages(data_subset1_rounded.iloc[:, 0], n)
rolling_avg_outlet = compute_rolling_averages(data_subset1_rounded.iloc[:, 1], n)


# Save the rolling average data to a CSV file
Rolling_avg_df = pd.DataFrame({
    'IN_LET': rolling_avg_inlet,
    'OUT_LET': rolling_avg_outlet
})
Rolling_avg_df.to_csv('Rolling_avg_interpolated_data_c1.csv', index=False)

print("interpolated data saved to 'Round_interpolated_data.csv'")


# Create a plot with dual x-axes for the first subset
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(data_subset1.iloc[:, 0], label='IN LET interpolated ', color='green', linestyle='-')
ax1.plot(data_subset1.iloc[:, 1], label='OUT LET interpolated', color= 'green', linestyle='-')
ax1.plot(data_subset1_rounded.iloc[:, 0], label='IN LET rounded ', color='red', linestyle='-')
ax1.plot(data_subset1_rounded.iloc[:, 1], label='OUT LET rounded', color= 'red', linestyle='-')
ax1.set_xlabel(' Data Index')
ax1.set_ylabel('Value')
ax1.legend(loc='upper left')
ax1.grid(True)


ax2 = ax1.twiny()
ax2.plot(rolling_avg_inlet, label='IN LET Rolling Avg ', color='blue', linestyle='--')
ax2.plot(rolling_avg_outlet, label='OUT LET Rolling Avg', color= 'blue', linestyle='--')
ax2.set_xlabel('Rolling Avg Data Index')
ax2.legend(loc='upper right')
ax2.grid(True)
plt.show()


