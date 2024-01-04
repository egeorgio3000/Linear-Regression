import matplotlib.pyplot as plt
import numpy as np

# Define your function f(x)
def f(x):
    return np.sin(x)  # Replace this with your actual function

# Generate x values
x = np.linspace(0, 10, 100)
# Calculate y values based on f(x)
y = f(x)

# Define the window size for the moving average
window_size = 5  # Adjust this as needed

# Calculate the moving average
smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='same')

# Create the plot with smoothed curve
plt.plot(x, smoothed_y, label='Smoothed Curve')
plt.plot(x, y, label='Original Curve', linestyle='--', alpha=0.5)  # Original curve for comparison
plt.xlabel('x')
plt.ylabel('y')
plt.title('Smoothed Curve')
plt.legend()
plt.show()