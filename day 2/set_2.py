

# Extract relevant columns (10, 11) and rows (from 3 to the end)
data_subset2 = data.iloc[2:, [10, 11]]

# Plot raw data for the second subset
plt.figure(figsize=(10, 6))
plt.plot(data_subset2.index, data_subset2.iloc[:, 0], label='IN LET', color='blue', linestyle='-')
plt.plot(data_subset2.index, data_subset2.iloc[:, 1], label='OUT LET', color='green', linestyle='-')
plt.title('Configuration 2 - Raw Data')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
# # Clean and interpolate data_subset2 from index 5300 to 5500 and 13500 to 13700

data_subset2.iloc[:, 0] = clean_and_interpolate(data_subset2.iloc[:, 0], 5300, 5500, 0, 500, 500)
data_subset2.iloc[:, 0] = clean_and_interpolate(data_subset2.iloc[:, 0], 13500, 13700, 0, 500, 500)
data_subset2.iloc[:, 1] = clean_and_interpolate(data_subset2.iloc[:, 1], 5300, 5500, 0, 500, 500)
data_subset2.iloc[:, 1] = clean_and_interpolate(data_subset2.iloc[:, 1], 13500, 13700, 0, 500, 500)

# Compute rolling averages for the second subset
rolling_avg_inlet2 = compute_rolling_averages(data_subset2.iloc[:, 0], n)
rolling_avg_outlet2 = compute_rolling_averages(data_subset2.iloc[:, 1], n)

# Create a plot with dual x-axes for the second subset
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot original data
ax1.plot(data_subset2.index, data_subset2.iloc[:, 0], label='IN LET', color='blue', linestyle='-')
ax1.plot(data_subset2.index, data_subset2.iloc[:, 1], label='OUT LET', color='green', linestyle='-')
ax1.set_xlabel('Original Data Index')
ax1.set_ylabel('Value')
ax1.legend(loc='upper left')
ax1.grid(True)

# Create a second x-axis for the rolling averages
ax2 = ax1.twiny()
ax2.plot(rolling_avg_inlet2.index * n, rolling_avg_inlet2, label='IN LET (Rolling Avg)', color='blue', linestyle='--')
ax2.plot(rolling_avg_outlet2.index * n, rolling_avg_outlet2, label='OUT LET (Rolling Avg)', color='green', linestyle='--')
ax2.set_xlabel('Rolling Average Index')
ax2.legend(loc='upper right')

plt.title('Configuration 2 with Dual X-Axis')
plt.show()