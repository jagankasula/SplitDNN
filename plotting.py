import pandas as pd
import matplotlib.pyplot as plt

# Read the data from Excel file
data = pd.read_excel('path_to_your_excel_file.xlsx')

# Extract the relevant columns
splits = data['split no.']
resnet34_times = data['resnet34']
resnet50_times = data['resnet50']
resnet101_times = data['resnet101']

# Convert time strings to datetime objects for plotting
resnet34_times = pd.to_datetime(resnet34_times, format='%M:%S.%f').dt.time
resnet50_times = pd.to_datetime(resnet50_times, format='%M:%S.%f').dt.time
resnet101_times = pd.to_datetime(resnet101_times, format='%M:%S.%f').dt.time

# Plot the times for each ResNet model
plt.plot(splits, resnet34_times, label='ResNet34')
plt.plot(splits, resnet50_times, label='ResNet50')
plt.plot(splits, resnet101_times, label='ResNet101')

# Set the plot labels and title
plt.xlabel('Split No.')
plt.ylabel('Time')
plt.title('Comparison of Times for ResNet Models')

# Add a legend
plt.legend()

# Show the plot
plt.show()