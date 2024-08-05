import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

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

# Equation to solve
def lambda_P_eq(P, H, lambda_P):
    term1 = 1 / (beta_STI_0 * (1 - m(H) * (1 - P)))
    term2 = C * (1 - m(H)) * beta_HIV * H * (1 - P)
    return (psi/(term1-(1-psi)/mu) - gamma - term2 - mu) - lambda_P * P

# Generate H values
H_values = np.linspace(0, 100, 100)

# Lambda values to consider
lambda_values = [1 / 360, 2 / 360, 3 / 360, 4 / 360]
lambda_labels = ["1 time per year", "2 times per year", "3 times per year", "4 times per year"]

# Plotting and solving
plt.figure(figsize=(10, 6))

for lambda_P, label in zip(lambda_values, lambda_labels):
    P_values = []
    for H in H_values:
        # Initial guess for P
        P_initial_guess = 0.5
        
        # Solve for P
        P_solution = fsolve(lambda P: lambda_P_eq(P, H, lambda_P), P_initial_guess)
        
        # Ensure P is in the range [0, 1]
        if 0 <= P_solution[0] <= 1:
            P_values.append(P_solution[0])
        else:
            P_values.append(np.nan)  # Mark as invalid
        
    plt.plot(H_values, P_values, label=label)

plt.xlabel('H')
plt.ylabel('P')
plt.title('Plot of (H, P) for different λ_P values')
plt.legend()
plt.grid(True)
plt.show()
