# Install necessary libraries (run in terminal or notebook if not already installed)
# !pip install numpy matplotlib scipy

#Step 0
    #Use NumPy for math and arrays
    #Use Matplotlib for drawing plots
import numpy as np
import matplotlib.pyplot as plt

#DATA
# Step 1.1: Load your points 
    # Split into x and y columns
data = np.loadtxt('Data.csv', delimiter=';', skiprows=1)

# Step 1.2: Fix your points 
    # Capture x_end from the original last point
    # Replace first point with (0, 0) without adding a new row
    # Replace last point with (x_end, 0)
x_end = np.max(data[:, 0])
data[0] = [0.0, 0.0]     # Set first point to (0,0)
data[-1] = [x_end, 0.0]  # Set last point to (x_end,0)

    # Remove duplicate x-values (keep first occurrence)
_, unique_indices = np.unique(data[:, 0], return_index=True)
data = data[unique_indices]
    # Sort by x to ensure interpolation works
data = data[data[:, 0].argsort()]  #x-values are cleanly ordered from left to right 
x_data = data[:, 0]
y_data = data[:, 1]

'''
    def print_points_matrix(data):
        print(f"{'Index':<10} {'X':<15} {'Y':<15}")
        print("-" * 40)
        for i, (x, y) in enumerate(data):
            print(f"{i:<10} {x:<15.4f} {y:<15.4f}")

    print_points_matrix(data)
'''
# TRIGONOMETRIC
# Step 2.1: Interpolate to uniform grid for Fourier input
x_uniform = np.linspace(x_data.min(), x_data.max(), len(x_data))
def linear_interpolation(x_data, y_data, x_uniform):
    y_uniform = np.zeros_like(x_uniform)
    j = 0
    for i in range(len(x_uniform)):
        while j < len(x_data) - 2 and x_uniform[i] > x_data[j + 1]: #j + 1 is always valid when we enter the loop.(-2 too othersite)
            j += 1
        x0, x1 = x_data[j], x_data[j + 1]
        y0, y1 = y_data[j], y_data[j + 1]
        t = (x_uniform[i] - x0) / (x1 - x0)
        y_uniform[i] = y0 + t * (y1 - y0)
    return y_uniform

y_uniform = linear_interpolation(x_data, y_data, x_uniform)

# Step 2.2: Compute Fourier-like frequencies
L = x_uniform.max() - x_uniform.min()
a1 = 1 * np.pi / L
a2 = 2 * np.pi / L
a3 = 3 * np.pi / L
a4 = 4 * np.pi / L
x_min = x_uniform.min()
x_max = x_uniform.max()

# Step 2.3: Scale x to [0, 2π]
x_trig = 2 * np.pi * (x_uniform - x_min) / (x_max - x_min)

# Step 3.2: Build the design matrix A 
def build_matrix(x, x_trig, k):
    A = np.column_stack([
        np.ones_like(x), x, x**2, x**3, x**4,  # Polynomial terms
        np.sin(a1 * x_trig),  # Term 1: w1 * sin(a1 x)
        np.cos(a2 * x_trig),  # Term 2: w2 * cos(a2 x)
        np.sin(a3 * x_trig),  # Term 3: w3 * sin(a3 x)
        np.cos(a4 * x_trig),  # Term 4: w4 * cos(a4 x)
        np.exp(k * x)         # Exponential term
    ])
    return A

# Step 4.1: Define Constraint Equations 
def enforce_constraints(x_last, y_last):
    # Constraint 1: f(0) = 0
    x0 = 0.0
    x0_trig = 2 * np.pi * (x0 - x_min) / (x_max - x_min)
    row1 = [
        1, x0, x0**2, x0**3, x0**4,               # Polynomial terms at x=0
        np.sin(a1 * x0_trig),                      # sin(a1 x_trig)
        np.cos(a2 * x0_trig),                      # cos(a2 x_trig)
        np.sin(a3 * x0_trig),                      # sin(a3 x_trig)
        np.cos(a4 * x0_trig),                      # cos(a4 x_trig)
        np.exp(k * x0)                             # Exponential term at x=0
    ]
    
    # Constraint 2: f(x_last) = y_last
    x_last_trig = 2 * np.pi * (x_last - x_min) / (x_max - x_min)
    row2 = [
        1, x_last, x_last**2, x_last**3, x_last**4,
        np.sin(a1 * x_last_trig),
        np.cos(a2 * x_last_trig),
        np.sin(a3 * x_last_trig),
        np.cos(a4 * x_last_trig),
        np.exp(k * x_last)
    ]
    
    C = np.array([row1, row2])  # Constraint matrix (2 rows)
    d = np.array([0, y_last])   # Constraint values
    return C, d


# Step 5: Solve the constrained least squares system
def solve_constrained_least_squares(A, y, C, d):
    m, n = A.shape
    KKT = np.block([
        [A.T @ A, C.T],
        [C, np.zeros((2, 2))]
    ])
    rhs = np.concatenate([A.T @ y, d])
    solution = np.linalg.solve(KKT, rhs)
    return solution[:n]

# Step 7: Search for best k manually
best_k = None
best_error = np.inf
best_theta = None
k_values = np.linspace(-3, 1, 200)

for k in k_values:
    try:
        A = build_matrix(x_uniform, x_trig, k)
        C, d = enforce_constraints(x_max, k)
        theta = solve_constrained_least_squares(A, y_uniform, C, d)
        y_fit = A @ theta
        error = np.sum((y_fit - y_uniform) ** 2)
        if error < best_error:
            best_error = error
            best_k = k
            best_theta = theta
    except np.linalg.LinAlgError:
        continue

# Step 8: Plot result
A_best = build_matrix(x_uniform, x_trig, best_k)
y_best = A_best @ best_theta

import matplotlib.pyplot as plt
plt.plot(x_uniform, y_uniform, 'o', label="Original (Uniform)")
plt.plot(x_uniform, y_best, '-', label=f"Fitted (k={best_k:.3f})")
plt.legend()
plt.grid(True)
plt.title("Manual Least Squares Fit (No Black Boxes)")
plt.show() 

# Step 9: Print coefficients
param_names = ['a0', 'a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4', 'c']
for name, value in zip(param_names, best_theta):
    print(f"{name} = {value:.5f}")
print(f"Best k = {best_k:.5f}")