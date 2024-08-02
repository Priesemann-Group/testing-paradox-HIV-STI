import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta_STI_0 = 0.0016 * 5
psi = 0.9
mu = 1.0 / 45.0 / 360.0
gamma = 1.0 / 1.32 / 360.0
C = 50
beta_HIV = 0.6341 / 360.0
m_max = 0.8
H_threshold = 0.1
m_epsilon = 0.01

# Mitigation function
def m(H):
    return m_max - (m_max / H_threshold) * m_epsilon * np.log(1 + np.exp((H_threshold - H) / m_epsilon))

# Compute lambda_P based on P and H
def compute_lambda_P(P, H):
    term1 = 1 / (beta_STI_0 * (1 - m(H) * (1 - P)))
    term2 = C * (1 - m(H)) * beta_HIV * H * (1 - P)
    lambda_P = (psi / (term1 - (1 - psi) / mu) - gamma - term2 - mu) / P
    return lambda_P

# Values for P and H
P_values = np.linspace(1, 100, 1000)  # Avoid exact 0 and 1
H_values = np.linspace(1, 100, 1000)

# Create meshgrid for P and H
P_grid, H_grid = np.meshgrid(P_values, H_values)
lambda_P_grid = np.zeros_like(P_grid)

# Compute lambda_P for each (P, H) pair
for i in range(P_grid.shape[0]):
    for j in range(P_grid.shape[1]):
        P = P_grid[i, j]
        H = H_grid[i, j]
        try:
            lambda_P_grid[i, j] = compute_lambda_P(P, H)
        except ZeroDivisionError:
            lambda_P_grid[i, j] = np.nan  # Handle potential division by zero

# Plotting
plt.figure(figsize=(10, 8))

# Heatmap Plot
heatmap = plt.contourf(P_grid, H_grid, lambda_P_grid, levels=50, cmap='viridis')
plt.colorbar(heatmap, label='lambda_P')
plt.xlabel('P')
plt.ylabel('H')
plt.title('Heatmap of lambda_P for P and H in [0,1]')
plt.show()

