#Step 0 LIBARARIES
import numpy as np
import matplotlib.pyplot as plt

# 1. DATA & SCALLING
# Step 1.1: Load your points 
    # Split into x and y columns
    #C:/Porsche_911/Data.csv
data = np.loadtxt('Data.csv', delimiter=';')

# Step 1.2: Fix points 
    # Capture x_end from the original last point
    # Replace first point with (0, 0) without adding a new row
    # Replace last point with (x_end, 0)
x_end = np.max(data[:, 0])
data[0] = [0.0, 0.0]     # Set first point to (0,0)
data[-1] = [x_end, 0.0]  # Set last point to (x_end,0)
    # Remove duplicate x-values & sort by x
# Manual duplicate removal
seen_x = {}
unique_data = []
for idx, row in enumerate(data):
    x = row[0]
    if x not in seen_x:
        seen_x[x] = True
        unique_data.append(row)
data = np.array(unique_data)

# Manual bubble sort by x
n = len(data)
for i in range(n):
    for j in range(0, n-i-1):
        if data[j, 0] > data[j+1, 0]:
            data[[j, j+1]] = data[[j+1, j]]


# Step 1.3: Dimension points 
real_length, real_height = 4.499, 1.279 # in meters
x_data = data[:, 0] * (real_length / np.max(data[:, 0]))
y_data = data[:, 1] * (real_height / np.max(data[:, 1]))
L = x_data.max()  # Total car length

# 2. TRIGONOMETRIC FREQUENCIES
# Step 2.1: Interpolate to uniform grid for Fourier input
def manual_binary_search(arr, x):
    low, high = 0, len(arr)
    while low < high:
        mid = (low + high) // 2
        if arr[mid] < x:
            low = mid + 1
        else:
            high = mid
    return low - 1



def safe_interpolation(x_raw, y_raw, num_points=50):
    x_uniform = np.linspace(0, L, num_points)
    y_uniform = np.empty_like(x_uniform)
    last_idx = len(x_raw) - 1

    for i, x in enumerate(x_uniform):
        idx = manual_binary_search(x_raw, x)
        idx = max(0, min(idx, last_idx - 1))
        x0, x1 = x_raw[idx], x_raw[idx + 1]
        y0, y1 = y_raw[idx], y_raw[idx + 1]
        if x1 == x0:
            y_uniform[i] = y0
        else:
            t = (x - x0) / (x1 - x0)
            y_uniform[i] = y0 + t * (y1 - y0)
    
    # Print results 
    DEBUG = False
    if DEBUG:
        print("\nInterpolation Debug:")
        print("x_uniform[:3] =", x_uniform[:3])
        print("x_uniform[-3:] =", x_uniform[-3:])
        print("y_uniform[:3] =", y_uniform[:3])
        print("y_uniform[-3:] =", y_uniform[-3:])   
    
    return x_uniform, y_uniform

# Step 2.2: Frequency Analysis
'''
    Finds optimal frequencies using direct trigonometric correlation
    Fourier Coefficient (sin term) = (2/N) * Σ y_i sin(a x_i)  
    Fourier Coefficient (cos term) = (2/N) * Σ y_i cos(a x_i)
    Score(a) = |∫ y(x) sin(a x) dx| + |∫ y(x) cos(a x) dx|
    where integral becomes sum of dots
    '''
def find_best_frequencies(x, y, L, num_freq=3):
    # Generate candidate frequencies (1-10 cycles over length L)
    candidate_cycles = np.linspace(0.5, 5, 100)
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
    # Print results 
    DEBUG = False
    if DEBUG:
        print("\nFrequency Analysis:")
        print("Selected Frequencies:")
        for i, a in enumerate(selected):
            print(f"  {i+1}. {a:.4f} rad/m")
    return np.array(selected)
x_uniform, y_uniform = safe_interpolation(x_data, y_data, num_points=50)
a_values = find_best_frequencies(x_uniform, y_uniform, L)

# 3. DESIGN MATRIX
def build_design_matrix(x, k, L, a_values):
    x_norm = x / L # normalize x to [0, 1]

    # Polynomial basis
    poly_terms = np.column_stack([
        np.ones_like(x_norm),  # a0
        x_norm,                # a1
        x_norm**2,             # a2
        x_norm**3              # a3
    ])
     # Trigonometric basis
    trig_terms = np.column_stack([
        f(a * x) for a in a_values for f in (np.sin, np.cos)]) # b1, b2, c1, c2, ...
    
    # Exponential term
    exp_term = np.exp(k * x_norm).reshape(-1, 1) # d1

    # Final design matrix
    A = np.hstack([poly_terms, trig_terms, exp_term])    
    return A
# Print results 
DEBUG = False
if DEBUG:
    test_k = 0.5
    A_test = build_design_matrix(x_uniform, test_k, L, a_values)
    print(f"\nDesign matrix built with k={test_k}")
    print("Matrix shape:", A_test.shape)  # should be (50, 4 + 2*len(a_values) + 1)
    print("First row:", A_test[0])

# 4. CONSTRAINT HANDLING
def build_constraints(k, L, a_values):
    # Constraint at x = 0
    x0_norm = 0.0
    row0_poly = [1.0, x0_norm, x0_norm**2, x0_norm**3]
    row0_trig = [np.sin(a * 0.0) for a in a_values] + [np.cos(a * 0.0) for a in a_values]
    row0_exp  = [np.exp(k * x0_norm)]
    row0 = row0_poly + row0_trig + row0_exp

    # Constraint at x = L
    xL_norm = 1.0
    rowL_poly = [1.0, xL_norm, xL_norm**2, xL_norm**3]
    rowL_trig = [np.sin(a * L) for a in a_values] + [np.cos(a * L) for a in a_values]
    rowL_exp  = [np.exp(k * xL_norm)]
    rowL = rowL_poly + rowL_trig + rowL_exp
    
    # Stack constraints
    C = np.array([row0, rowL])
    d = np.array([0.0, 0.0])
# Print results 
    DEBUG = False
    if DEBUG:
        print("\nConstraint Matrix Debug:")
        print("C shape:", C.shape)
        print("C[0] (x=0):", np.round(C[0], 3))
        print("C[1] (x=L):", np.round(C[1], 3))
        print("Target values d:", d)

    return C, d


# 5. LEAST SQUARES 
def manual_solve(A, b):
    """Solves Ax = b with manual Gaussian elimination and pivoting."""
    n = A.shape[0]
    aug = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for i in range(n):
        # Manual argmax for pivoting
        max_row = i
        for j in range(i, n):
            if abs(aug[j, i]) > abs(aug[max_row, i]):
                max_row = j
        aug[[i, max_row]] = aug[[max_row, i]]

        # Eliminate below
        for j in range(i+1, n):
            factor = aug[j, i] / aug[i, i]
            aug[j, i:] -= factor * aug[i, i:]

    # Back-substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (aug[i, -1] - np.sum(aug[i, i+1:n] * x[i+1:n])) / aug[i, i]
    return x

def constrained_regression(A, y, C, d, alpha=1e-4):
    """Solve (AᵀA + αI)θ = Aᵀy with equality constraints Cθ = d"""
    n = A.shape[1]
    # Regularized left-hand side (AᵀA + αI)
    lhs = A.T @ A + alpha * np.eye(n)
    rhs = A.T @ y

    top = np.hstack([lhs, C.T])
    bottom = np.hstack([C, np.zeros((C.shape[0], C.shape[0]))])
    KKT = np.vstack([top, bottom])
    rhs_full = np.concatenate([rhs, d])

    # Solve system
    try:
        sol = manual_solve(KKT, rhs_full)  # <-- Changed from np.linalg.solve
        
        theta = sol[:n]

        print("\n[Strict Constraint Fit]")
        print(f"  f(0)  = {C[0] @ theta:.2e}")
        print(f"  f(L)  = {C[1] @ theta:.2e}")
        print(f"  Residual norm = {np.linalg.norm(A @ theta - y):.5f}")

        return theta
    except np.linalg.LinAlgError as e:
        print("Manual solve failed:", e)
        raise

# 6. OPTIMIZE k
best_params = None
min_error = float('inf')
k_candidates = np.linspace(-2, 2, 300)  # Wider search range

DEBUG = False  

for k in k_candidates:
    try:
        # Build matrices
        A = build_design_matrix(x_uniform, k, L, a_values)
        C, d = build_constraints(k, L, a_values)        
        # Solve with constraints
        theta = constrained_regression(A, y_uniform, C, d)
        # Calculate constrained error
        y_fit = A @ theta
        error = np.mean((y_fit - y_uniform)**2)
       
        if DEBUG and k % 0.5 == 0:  # print every 0.5 step
            print(f"\nk = {k:.2f} | MSE: {error:.5f}")

        if error < min_error:
            min_error = error
            best_params = (k, theta)

    except np.linalg.LinAlgError as e:
        print(f"[WARNING] LinAlgError at k={k:.2f} → {e}")
        continue

if best_params is None:
    raise RuntimeError("No valid solution found — try adjusting model or k range.")

best_k, best_theta = best_params

if DEBUG:
    print("\nBest k found:")
    print(f"best_k = {best_k:.4f}")
    print(f"min_error = {min_error:.6f}")
    print(f"theta[:5] = {np.round(best_theta[:5], 4)}")

# 7. ROMBERG INTEGRATION FOR AREA
def romberg_integration(f, a, b, max_steps=6):
    """Manual implementation of Romberg integration without black-box functions"""
    R = np.zeros((max_steps, max_steps))
    h = b - a
    # Initial trapezoidal rule (just endpoints)
    R[0, 0] = 0.5 * h * (f(a) + f(b))

    for i in range(1, max_steps):
        h /= 2  # New step size
        # Calculate new interior points
        total = 0.0
        num_points = 2**(i-1)
        for k in range(1, num_points + 1):
            x = a + (2*k - 1)*h
            total += f(x)
        # Update trapezoidal estimate
        R[i, 0] = 0.5 * R[i-1, 0] + h * total
        
        # Richardson extrapolation
        for j in range(1, i+1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1])/(4**j - 1)
    
    return R[-1, -1]

# Define the fitted function using obtained parameters
def car_profile(x):
    """Evaluates f(x) using the optimized parameters"""
    x_array = np.array([x])
    A = build_design_matrix(x_array, best_k, L, a_values)
    return (A @ best_theta)[0]

# Calculate area
# Calculate area with Romberg (using 5 and 6 steps for error estimate)
area_5 = romberg_integration(car_profile, 0, L, max_steps=5)
area_6 = romberg_integration(car_profile, 0, L, max_steps=6)
area = area_6  # Use the more accurate result
error_estimate = abs(area_6 - area_5)  # Difference between steps 5 and 6

print("\nArea Calculation Results:")
print(f"Area under fitted curve: {area:.6f} m²")
print(f"Romberg relative error estimate: {abs(area - romberg_integration(car_profile, 0, L, 6)):.2e}")

# Step 8: PLOT
C, d = build_constraints(best_k, L, a_values)
x_fine = np.linspace(0, L, 100)
A_fine = build_design_matrix(x_fine, best_k, L, a_values)
y_fit = A_fine @ best_theta

# Explicitly set first and last points to exact constraints
y_fit[0] = 0.0   # Force f(0) = 0
y_fit[-1] = 0.0  # Force f(L) = 0

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(x_data, y_data, 'ro', markersize=4, label='Original Data')
plt.plot(x_fine, y_fit, 'b-', linewidth=2, label=f'Fit (k={best_k:.2f})')
plt.plot([0, L], [0, 0], 'kx', markersize=10, label='Constraints')

plt.fill_between(x_fine, y_fit, alpha=0.2, color='cyan', label='Area')
plt.plot([], [], ' ',  # Invisible marker
         label=f'Area: {area:.3f} m²\nError: {error_estimate:.1e}')

plt.title('Car Profile Approximation', fontsize=14)
plt.xlabel('Length (m)', fontsize=14)
plt.ylabel('Height (m)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis('equal')  # Ensure realistic car shape regardless of window size
plt.tight_layout()
plt.show()

# OUTPUT RESULTS
print("\nOptimization Results:")
print(f"Best k: {best_k:.4f}")
print(f"Polynomial Coefficients: {np.round(best_theta[:4], 4)}")
for i, a in enumerate(a_values):
    sin_coeff = best_theta[4 + 2*i]
    cos_coeff = best_theta[5 + 2*i]
    print(f"  Frequency {i+1} ({a:.4f} rad/m): sin = {sin_coeff:.4f}, cos = {cos_coeff:.4f}")
print(f"Exponential Coefficient: {best_theta[-1]:.4f}")
print(f"Mean Squared Error: {min_error:.6f}")

# Check constraints
print("\nConstraint Verification:")
print(f"f(0.000 m) = {A_fine[0] @ best_theta:.2e}")
print(f"f({L:.3f} m) = {A_fine[-1] @ best_theta:.2e}")
print("C @ theta =", C @ best_theta)

