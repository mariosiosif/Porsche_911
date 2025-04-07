# 0. LIBARARIES
import numpy as np
import matplotlib.pyplot as plt

# 1. DATA & SCALLING & UNIFORMITY
# Step 1.1: Load your points 
data = np.loadtxt('Data.csv', delimiter=';')

# Step 1.2: Fix points 
x_end = np.max(data[:, 0])
data[0] = [0.0, 0.0]     # Set first point to (0,0)
data[-1] = [x_end, 0.0]  # Set last point to (x_end,0)

# Step 1.3 Duplicate removal
x_values = data[:, 0]
if len(np.unique(x_values)) != len(x_values):
    raise ValueError("Error: Duplicate x-values found — y(x) is not a function.")

# Step 1.4 Sort by x
n = len(data)
for i in range(n):
    for j in range(0, n-i-1):
        if data[j, 0] > data[j+1, 0]:
            data[[j, j+1]] = data[[j+1, j]]

# Step 1.5: Dimension points 
real_length, real_height = 4.499, 1.279 # in meters
x_data = data[:, 0] * (real_length / np.max(data[:, 0]))
y_data = data[:, 1] * (real_height / np.max(data[:, 1]))
L = np.max(x_data)  # Total car length

# Step 1.6: Uniformity of points using Lagrange interpolation to num_points evenly spaced in [0, L]
def interpolate_uniform(x_raw, y_raw, L, num_points):
    if num_points is None:
        num_points = len(x_raw)

    # Create uniform x-values
    x_uniform = np.linspace(0, L, num_points)
    y_uniform = np.empty_like(x_uniform)

    # Loop and interpolate
    for i, x in enumerate(x_uniform):
        idx = np.searchsorted(x_raw, x) - 1
        x0, x1 = x_raw[idx], x_raw[idx + 1]
        y0, y1 = y_raw[idx], y_raw[idx + 1]
        y_uniform[i] = y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    x_uniform[-1] = x_raw[-1]
    y_uniform[-1] = y_raw[-1]
    x_uniform[0] = x_raw[0]
    y_uniform[0] = y_raw[0]

    return x_uniform, y_uniform
num_points = len(x_data)  # same number of points as input
x_uniform, y_uniform = interpolate_uniform(x_data, y_data, L, num_points)

# 3. TRIGONOMETRIC FREQUENCIES
# Step 3.1: Frequency Analysis
'''
    Finds optimal frequencies using direct trigonometric correlation
    Fourier Coefficient (sin term) = (2/N) * Σ y_i sin(a x_i)  
    Fourier Coefficient (cos term) = (2/N) * Σ y_i cos(a x_i)
    Score(a) = |∫ y(x) sin(a x) dx| + |∫ y(x) cos(a x) dx|
    where integral becomes sum of dots
    '''
def find_best_frequencies(x, y, L, num_freq=3):
    # Generate candidate frequencies (1-10 cycles over length L)
    candidate_cycles = np.linspace(0.5, 5, 200)
    candidates = 2 * np.pi * candidate_cycles / L
    scores = np.zeros_like(candidates)  # Score each candidate frequency

    for i, a in enumerate(candidates):
        scores[i] = (
        np.abs(np.sum(y * np.sin(a * x))) + 
        np.abs(np.sum(y * np.cos(a * x)))
        )

    top_indices = np.argsort(-scores)   #  best spaced frequencies
    selected = []
    for idx in top_indices:
        if all(np.abs(candidates[idx] - s) > (2 * np.pi / L) for s in selected):
            selected.append(candidates[idx])
            if len(selected) == num_freq:
                break
    return np.array(selected)
a_values = find_best_frequencies(x_uniform, y_uniform, L)

# 3. Manual Gaussian elimination with pivoting
def gauss_elimination(A, b):
    n = len(b)
    aug = np.hstack([A, b.reshape(-1,1)])
    
    for i in range(n):
        # Partial pivoting
        max_row = np.argmax(np.abs(aug[i:n, i])) + i
        aug[[i, max_row]] = aug[[max_row, i]]
        
        # Eliminate
        for j in range(i+1, n):
            factor = aug[j,i] / aug[i,i]
            aug[j,i:] -= factor * aug[i,i:]
    
    # Back-substitute
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (aug[i,-1] - np.dot(aug[i,i+1:n], x[i+1:n])) / aug[i,i]
    return x

# Build normal equations manually
def manual_least_squares(x_points, y_points, basis_funcs):
    n_basis = len(basis_funcs)
    ATA = np.zeros((n_basis, n_basis))
    ATb = np.zeros(n_basis)
    
    # Compute ATA and ATb
    for i in range(n_basis):
        phi_i = basis_funcs[i](x_points)
        ATb[i] = np.sum(y_points * phi_i)
        for j in range(i, n_basis):
            phi_j = basis_funcs[j](x_points)
            ATA[i,j] = ATA[j,i] = np.sum(phi_i * phi_j)
    
    # Solve using manual Gaussian elimination
    coeffs = gauss_elimination(ATA, ATb)
    return coeffs

# Apply constraints by modifying basis functions
def constrained_basis(L, a_values, k):
    """
    Builds basis functions with:
    - 3 polynomial terms
    - 3 trig terms with different frequencies (sin, cos, sin²)
    - 1 exponential term
    All terms vanish at x = 0 and x = L.
    """
    a1, a2, a3 = a_values

    return [
        # Polynomial terms (vanishing at boundaries)
        lambda x: (x / L) - (x / L)**3,
        lambda x: (x / L)**2 - (x / L)**3,
        lambda x: (x / L)**4 - (x / L)**3,

        # Trig terms with different frequencies
        lambda x: np.sin(a1 * x) * (x * (L - x) / L**2),
        lambda x: np.cos(a2 * x) * (x * (L - x) / L**2),
        lambda x: (np.sin(a3 * x))**2 * (x * (L - x) / L**2),

        # Exponential term (also vanishing at both ends)
        lambda x: (np.exp(k * x / L) - (x / L) * np.exp(k) - (1 - x / L))
    ]

# 4. Optimization and model evaluation
def evaluate_model(x, basis, coeffs):
    return sum(c * f(x) for c, f in zip(coeffs, basis))
# Optimization loop for k (complete implementation)
best_k = None
best_coeffs = None
min_error = float('inf')
x_fine = np.linspace(0, L, 100)  # For final plotting

for k in np.linspace(-2, 2, 50):
    try:
        # Create constrained basis
        basis = constrained_basis(L, a_values, k)
        
        # Solve least squares
        coeffs = manual_least_squares(x_uniform, y_uniform, basis)
        
        # Calculate error
        y_fit = evaluate_model(x_uniform, basis, coeffs)
        current_error = np.mean((y_fit - y_uniform)**2)
        
        # Update best parameters
        if current_error < min_error:
            min_error = current_error
            best_k = k
            best_coeffs = coeffs
    except Exception as e:
        print(f"Skipping k={k:.2f} due to error: {str(e)}")
        continue



# 5. Final model evaluation and area calculation
# Generate fine grid for plotting
final_basis = constrained_basis(L, a_values, best_k)
y_fine = evaluate_model(x_fine, final_basis, best_coeffs)

# Force exact boundary conditions (numerical safety)
y_fine[0] = 0.0
y_fine[-1] = 0.0

# Area calculation using trapezoidal rule
area = np.trapz(y_fine, x_fine)

# 6. Plotting (corrected)
plt.figure(figsize=(12, 6))
plt.plot(x_data, y_data, 'ro', markersize=4, label='Original Data')
plt.plot(x_fine, y_fine, 'b-', linewidth=2, label=f'Best Fit (k={best_k:.2f})')
plt.fill_between(x_fine, y_fine, alpha=0.2, color='cyan', label=f'Area: {area:.3f} m²')

plt.title('Car Profile Approximation')
plt.xlabel('Length (m)')
plt.ylabel('Height (m)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()

# 7. Results output (corrected)
print("\nOptimization Results:")
print(f"Best k: {best_k:.4f}")
print("Coefficients:")
for i, coeff in enumerate(best_coeffs):
    print(f"  Basis {i+1}: {coeff:.4e}")

# Constraint verification
print("\nConstraint Verification:")
print(f"f(0) = {evaluate_model(np.array([0.0]), final_basis, best_coeffs)[0]:.2e}")
print(f"f({L:.3f}) = {evaluate_model(np.array([L]), final_basis, best_coeffs)[0]:.2e}")

plt.figure(figsize=(10, 5))
plt.plot(x_data, y_data, 'ro-', label='Original Data (from WebPlotDigitizer)')
plt.plot(x_uniform, y_uniform, 'bo--', label='Interpolated Uniform Data')

plt.title("Original vs Interpolated Uniform Data")
plt.xlabel("Length (m)")
plt.ylabel("Height (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()