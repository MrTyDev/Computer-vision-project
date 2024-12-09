import matplotlib.pyplot as plt
import numpy as np
import json

# Load the JSON data
with open('faces_data.json', 'r') as f:
    faces_data = json.load(f)

# Extract x and y coordinates
data = faces_data[0]
x = [point[1] for point in data]
y = [point[2] for point in data]

# Plot the data
plt.scatter(x, y)

# Display the plot
plt.show()