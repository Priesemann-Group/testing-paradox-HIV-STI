import numpy as np
import matplotlib.pyplot as plt

# Parameters
gamma = 1 / 1.32
tgamma = 1 / 7
psi = 0.9
lambda_o = 0.3 / 11
lambda_h = 0.3 / 11
mu = 1 / 45

# Calculating a and b
a = lambda_o + mu
b = gamma + lambda_h + mu

# Define a function to calculate R0
def calculate_R0(alpha):
    return a / (psi * alpha * (a / b) + (1 - psi) * alpha)

# Define a function to calculate I_star
def calculate_I_star(alpha):
    R0 = calculate_R0(alpha)
    numerator = (mu + tgamma) * (R0 - 1)
    denominator = -b / psi + gamma - tgamma * (1 + b * (1 - psi) / (a * psi))
    I_star = numerator / denominator
    return max(I_star, 0)  # Ensuring I_star is non-negative

# Find the alpha value where R0 = 1
alpha_bifurcation = (lambda_o + mu) / (psi * (lambda_o + mu) / b + (1 - psi))

# Calculate the critical points
alpha_C_DF = a + b
alpha_det = (-a*b)/(b*(psi-1)-a*psi)

# Range for alpha
alpha_values = np.linspace(0.1, 1, 500)
I_star_values = np.array([calculate_I_star(alpha) for alpha in alpha_values])

# Plotting
plt.plot(alpha_values, I_star_values, label='$I^*$', color='blue')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--', label='Disease-free equilibrium')
plt.axvline(alpha_bifurcation, color='red', linestyle='--', label='$\\alpha_c$ where $R_0=1$')
plt.axvline(alpha_det, color='purple', linestyle='--', label='$\\alpha_{\text{det}}$')

# Annotating the bifurcation, disease-free equilibrium, and detection threshold points
plt.text(alpha_bifurcation, max(I_star_values) * 0.6, f'$\\alpha_c = {alpha_bifurcation:.4f}$',
         color='red', fontsize=10, verticalalignment='center')
plt.text(alpha_det, max(I_star_values) * 0.4, f'$\\alpha_{{det}} = {alpha_det:.4f}$',
         color='purple', fontsize=10, verticalalignment='center')

plt.xlabel('$\\alpha$')
plt.ylabel('$I^*$')
plt.title('Asymptomatic Infections $I^*$ vs. Parameter $\\alpha$ with $\\psi=0.9$')
plt.legend()
plt.grid(True)
plt.show()
