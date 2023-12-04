import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Get the directory of the current script
directory = os.path.dirname(os.path.realpath(__file__))

# Initialize an empty list to store all similarity values
all_similarity_values = []

# Iterate over each file in the directory
for file in os.listdir(directory):
    if file.endswith('.csv'):
        # Read the CSV file
        filepath = os.path.join(directory, file)
        df = pd.read_csv(filepath)

        # Check if 'Similarity' column exists
        if 'Similarity' in df.columns:
            # Append the values to the list
            all_similarity_values.extend(df['Similarity'].tolist())

# Convert the list to a pandas series
similarity_series = pd.Series(all_similarity_values)

# Generate a distribution plot
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(similarity_series, bins=30, edgecolor='black')
plt.title('Distribution of Similarity Values Across All CSV Files')
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.grid(True)

# Calculate the cumulative distribution
cumulative_counts = np.cumsum(counts)

# Find the bins where the cumulative count is approximately 1/3 and 2/3 of the total
total_count = len(similarity_series)
one_third_bin = bins[np.where(cumulative_counts >= total_count / 3)[0][0]]
two_third_bin = bins[np.where(cumulative_counts >= 2 * total_count / 3)[0][0]]

# Draw vertical lines at these bins
plt.axvline(x=one_third_bin, color='r', linestyle='dashed', linewidth=2)
plt.axvline(x=two_third_bin, color='r', linestyle='dashed', linewidth=2)

# Show the plot
plt.show()

# Print the x values for the two lines
print("X value for the first line (1/3 of total):", one_third_bin)
print("X value for the second line (2/3 of total):", two_third_bin)
