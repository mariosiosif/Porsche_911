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

# 2. TRIGONOMETRIC FREQUENCIES ANALYSIS
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

# 3. LEAST SQUARE REGRESSION
# Step 3.1 Manual Gaussian elimination with pivoting
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

# Step 3.2 Build normal equations 
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

# 4. CONSTRAINT HANDLING
def constrained_basis(L, a_values, k):
    """Creates basis functions that automatically satisfy f(0)=0 and f(L)=0"""
    basis = [
        # Polynomial terms (3-4 terms)
        lambda x: (x/L) - (x/L)**3,
        lambda x: (x/L)**2 - (x/L)**3,
        lambda x: (x/L)**4 - (x/L)**3,
        
        # Trigonometric terms (3-4 terms)
        *[lambda x, a=a: np.sin(a*x) * x*(L-x)/L**2 for a in a_values],
        *[lambda x, a=a: np.cos(a*x) * x*(L-x)/L**2 for a in a_values[:2]],  # Use first 2 frequencies
        
        # Exponential term (enforced to zero at boundaries)
        lambda x: (np.exp(k*x/L) - (x/L)*np.exp(k) - (1 - x/L))
    ]
    return basis

# 5. OPTIMIZATION AND CALCULATIONS
def evaluate_model(x, basis, coeffs):
    return sum(c * f(x) for c, f in zip(coeffs, basis))
# Optimization loop for k (complete implementation)
best_k = None
best_coeffs = None
min_error = float('inf')
x_fine = np.linspace(0, L, 100)  # For final plotting

for k in np.linspace(-3, 3, 100):
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



# 6. FINAL MODEL
# Generate fine grid for plotting
final_basis = constrained_basis(L, a_values, best_k)
y_fine = evaluate_model(x_fine, final_basis, best_coeffs)

# Force exact boundary conditions (numerical safety)
y_fine[0] = 0.0
y_fine[-1] = 0.0


# 7. ROMBERG INTEGRATION FOR AREA
def romberg_integration(f, a, b, max_steps=6):
    """Manual implementation of Romberg integration without black-box functions"""
    R = np.zeros((max_steps, max_steps))
    h = b - a
    # Initial trapezoidal rule (just endpoints)
    R[0, 0] = 0.5 * h * (f(a)[0] + f(b)[0])

    for i in range(1, max_steps):
        h /= 2  # New step size
        # Calculate new interior points
        total = 0.0
        num_points = 2**(i-1)
        for k in range(1, num_points + 1):
            x = a + (2*k - 1)*h
            total += f(x)[0]
        # Update trapezoidal estimate
        R[i, 0] = 0.5 * R[i-1, 0] + h * total
        
        # Richardson extrapolation
        for j in range(1, i+1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1])/(4**j - 1)
    
    return R[-1, -1]

# Define fitted curve using final basis + best coefficients
def car_profile(x):
    """Evaluates f(x) using the optimized basis and coefficients"""
    x = np.array([x]) if np.isscalar(x) else np.array(x)
    basis = constrained_basis(L, a_values, best_k)
    return np.sum([c * f(x) for c, f in zip(best_coeffs, basis)], axis=0)

# Romberg area calculation
area_5 = romberg_integration(car_profile, 0, L, max_steps=5)
area_6 = romberg_integration(car_profile, 0, L, max_steps=6)
area = area_6
error_estimate = abs(area_6 - area_5)

# 8. PLOT & PRINT
# Step 8.1 Plot
# Plot the fitted car curve
x_fine = np.linspace(0, L, 200)
y_fit = car_profile(x_fine)

# Enforce exact boundary constraints
y_fit[0] = 0.0
y_fit[-1] = 0.0

plt.figure(figsize=(12, 6))
plt.plot(x_data, y_data, 'ro', markersize=4, label='Original Data')
plt.plot(x_fine, y_fit, 'b-', linewidth=2, label=f'Fitted Curve (k={best_k:.2f})')
plt.plot([0, L], [0, 0], 'kx', markersize=10, label='Boundary Constraints')
plt.fill_between(x_fine, y_fit, alpha=0.2, color='cyan', label=f'Area ≈ {area:.3f} m²')

plt.title('Car Profile Approximation', fontsize=14)
plt.xlabel('Length (m)')
plt.ylabel('Height (m)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()

# Step 8.2 Print
# Constraint verification
print("\nConstraint Verification:")
print(f"f(0) = {evaluate_model(np.array([0.0]), final_basis, best_coeffs)[0]:.2e}")
print(f"f({L:.3f}) = {evaluate_model(np.array([L]), final_basis, best_coeffs)[0]:.2e}")

# Optimization info
print("\nOptimization Results:")
print(f"Best k: {best_k:.4f}")
print("Coefficients:")
for i, coeff in enumerate(best_coeffs):
    print(f"  Basis {i+1}: {coeff:.4e}")


# Area
print("\nArea Calculation Results:")
print(f"Area under fitted curve: {area:.6f} m²")
print(f"Romberg relative error estimate: {error_estimate:.2e}")



