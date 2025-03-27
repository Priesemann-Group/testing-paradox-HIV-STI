import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import sys
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

sys.path.append("../src/")
from tapm import utils

jax.config.update("jax_enable_x64", True)

args, y0 = utils.read_params(filename="model_params_HIVandSTI.txt")
lambdaPs = [1/360, 2/360, 4/360] 
betaSTIs = [0.0016*5, 0.0016*7]  

# here you have to choose what you want
onlyplots = True # if True, only plots are generated, if False, first data is generated and then plots
derivative = "dP" # dP: derivative with respect to PrEP adoption, dH: derivative with respect to risk awareness
prevalence = True # if True, prevalence is calculated, if False, incidence is calculated

# helper stuff, don't change
if derivative == "dP":
    argnums = 0
else:
    argnums = 1
if prevalence:
    prevorinc = "prevalence"
else:
    prevorinc = "incidence"

if not onlyplots:
    for lambdaP in lambdaPs:
        for betaSTI in betaSTIs:
            if prevalence:
                def Nobs(P,H):
                    return (lambdas(P,H) * Is(P,H) + lambdaa(P,H) * Ia(P,H))/args["gammaT_STI"] # prevalence

                def Nreal(P,H):
                    return Ia(P,H) + Is(P,H) # prevalence
            else:
                def Nobs(P,H):
                    return lambdas(P,H) * Is(P,H) + lambdaa(P,H) * Ia(P,H) # incidence
                def Nreal(P,H):
                    return betaSTI *((1-m(P,H))*(1-P) + P) * S(P,H) * (Ia(P,H) + Is(P,H)) + args['Sigma'] # incidence


            def S(P,H):
                gamma, tilde_gamma, mu, psi, Sigma = args["gamma_STI"], args["gammaT_STI"], args["mu"], args["psi"], args["Sigma"]
                kappa_val = kappa(P, H)
                lambda_s_val = lambdas(P, H)
                lambda_a_val = lambdaa(P, H)

                C = (gamma + lambda_a_val + mu) * (1 - psi) / (psi * (lambda_s_val + mu))
                A = kappa_val * (1 + C) * (-(lambda_s_val + mu) * C + (gamma - tilde_gamma * (1 + C)) * (1 - psi))
                B = (tilde_gamma + mu) * ((1 - psi) * kappa_val * (1 + C) - (lambda_s_val + mu) * C)
                D = (tilde_gamma + mu) * (1 - psi) * Sigma

                discriminant = B**2 - 4 * A * D

                Ia_star = (-B - jnp.sqrt(discriminant)) / (2 * A)
                S_star = ((lambda_s_val + mu) * C * Ia_star - (1 - psi) * Sigma) / ((1 - psi) * kappa_val * (1 + C) * Ia_star)

                return S_star

            def Ia(P,H):
                gamma, tilde_gamma, mu, psi, Sigma = args["gamma_STI"], args["gammaT_STI"], args["mu"], args["psi"], args["Sigma"]
                kappa_val = kappa(P, H)
                lambda_s_val = lambdas(P, H)
                lambda_a_val = lambdaa(P, H)

                C = (gamma + lambda_a_val + mu) * (1 - psi) / (psi * (lambda_s_val + mu))
                A = kappa_val * (1 + C) * (-(lambda_s_val + mu) * C + (gamma - tilde_gamma * (1 + C)) * (1 - psi))
                B = (tilde_gamma + mu) * ((1 - psi) * kappa_val * (1 + C) - (lambda_s_val + mu) * C)
                D = (tilde_gamma + mu) * (1 - psi) * Sigma

                discriminant = B**2 - 4 * A * D

                Ia_star = (-B - jnp.sqrt(discriminant)) / (2 * A)

                return Ia_star

            def Is(P,H):
                gamma, tilde_gamma, mu, psi, Sigma = args["gamma_STI"], args["gammaT_STI"], args["mu"], args["psi"], args["Sigma"]
                kappa_val = kappa(P, H)
                lambda_s_val = lambdas(P, H)
                lambda_a_val = lambdaa(P, H)

                C = (gamma + lambda_a_val + mu) * (1 - psi) / (psi * (lambda_s_val + mu))
                A = kappa_val * (1 + C) * (-(lambda_s_val + mu) * C + (gamma - tilde_gamma * (1 + C)) * (1 - psi))
                B = (tilde_gamma + mu) * ((1 - psi) * kappa_val * (1 + C) - (lambda_s_val + mu) * C)
                D = (tilde_gamma + mu) * (1 - psi) * Sigma

                discriminant = B**2 - 4 * A * D

                Ia_star = (-B - jnp.sqrt(discriminant)) / (2 * A)
                Is_star = C * Ia_star

                return Is_star



            def kappa(P, H):
                return betaSTI * (1 - m(P, H) * (1 - P))

            def lambdas(P, H):
                return args["lambda_s"] + lambdaa(P, H)

            def lambdaa(P, H):
                return lambdaH(P, H) * (1 - P) + lambdaP * P

            def lambdaH(P, H):
                return args["c"] * (1 - m(P, H)) * args["beta_HIV"] * H

            def m(P, H):
                return args["min_exp"] + (args["max_exp"] - args["min_exp"]) * (1 - jnp.exp(-H / args["tau_exp"]))

            def R0(P, H):
                return args["psi"] * (betaSTI * (1 - m(P, H) * (1 - P))) / (args["gamma_STI"] + lambdaa(P, H) + args["mu"]) + (1 - args["psi"]) * (betaSTI * (1 - m(P, H) * (1 - P))) / (lambdas(P, H) + args["mu"])


            dNobs_dP_jax =jax.grad(Nobs, argnums=argnums)
            dNreal_dP_jax = jax.grad(Nreal, argnums=argnums)


            # compare gradients
            Ps = np.linspace(0, 1, 201)
            Hs = np.linspace(0, 0.2, 201)

            dNobsdP = np.zeros((len(Hs), len(Ps)))
            dNrealdP = np.zeros((len(Hs), len(Ps)))

            for i, H in enumerate(Hs):
                for j, P in enumerate(Ps):
                    dNobsdP[i, j] = dNobs_dP_jax(P, H)
                    dNrealdP[i, j] = dNreal_dP_jax(P, H)

            # compare Nobs and Nrea, not gradients
            Nobs_vals = np.zeros((len(Hs), len(Ps)))
            Nreal_vals = np.zeros((len(Hs), len(Ps)))
            Nobs_Nreal_vals_comparison = np.zeros((len(Hs), len(Ps)))

            for i, H in enumerate(Hs):
                for j, P in enumerate(Ps):
                    Nobs_vals[i, j] = Nobs(P, H)
                    Nreal_vals[i, j] = Nreal(P, H)
                    Nobs_Nreal_vals_comparison[i, j] = Nobs_vals[i, j] - Nreal_vals[i, j]

            # save stuff as npy files

            with open(
                "../results/Nreal_Nobs_values_and_derivatives_lambdap%g_betaSTI%g_%s_dN%s.npy"
                % (lambdaP * 360, betaSTI, prevorinc, derivative),
                "wb",
            ) as f:
                np.save(f, Ps)
                np.save(f, Hs)
                np.save(f, dNobsdP)
                np.save(f, dNrealdP)
                np.save(f, Nobs_vals)
                np.save(f, Nreal_vals)


# Plotting----------------------------------------------------------------------------------------------------------------------------
Hs = np.linspace(0, 0.2, 201)
Ps = np.linspace(0, 1.0, 201)

lambda_P_values = [4 / 360.0, 2 / 360.0, 1 / 360.0]
lambda_P_labels = ["4/year", "2/year", "1/year"]
beta_STI_values = [0.0016 * 5.0, 0.0016 * 7.0]
beta_STI_labels = ["mid", "high"]
colors_stability = ["#E9002D", "#FFAA00", "#00B000"]

# Plot set-up
fig = plt.figure(figsize=(8.0, 2.5))
outer_grid = fig.add_gridspec(1, 2, width_ratios=[0.8, 2.5], wspace=0.7)

# Left side of the plot, Nobs, Nreal, Nreal-Nobs---------------------------------------------------------------------------------------------------
# set-up
left_grid = outer_grid[0].subgridspec(3, 1, hspace=0.7)
axs = []
for i in range(3):
    ax = fig.add_subplot(left_grid[i])
    axs.append(ax)
#cmap = plt.get_cmap("viridis")
def discretize_cmaps(cmap, N):
    cmap = plt.colormaps[cmap]
    colors = cmap(np.linspace(0, 1, N))
    res = ListedColormap(colors)
    res.set_bad("#ABABAB")
    return res
cmap = discretize_cmaps("viridis", 15)

# load data
with open("../results/Nreal_Nobs_values_and_derivatives_lambdap%g_betaSTI%g_%s_dN%s.npy" % (1, 0.008, prevorinc, derivative),"rb") as f:
    Ps_plot = np.load(f)
    Hs_plot = np.load(f)
    dNobsdP_plot = np.load(f)
    #dNobsdH_plot = np.load(f)
    dNrealdP_plot = np.load(f)
    #dNrealdH_plot = np.load(f)
    Nobs_plot = np.load(f) * 100000
    Nreal_plot = np.load(f) * 100000


Nreal_minus_Nobs_plot = Nreal_plot - Nobs_plot
threshold_setbad = 0.00003 * 100000
Nobs_plot[Nobs_plot < threshold_setbad] = np.nan
Nreal_plot[Nreal_plot < threshold_setbad] = np.nan
Nreal_minus_Nobs_plot[Nreal_minus_Nobs_plot < threshold_setbad] = np.nan
results = [Nreal_plot, Nobs_plot, Nreal_minus_Nobs_plot]
titles = ["Real cases", "Observed cases", "Undetected cases"]

# plotting case numbers
for i, ax in enumerate(axs):
    for color in colors_stability:
        cax = ax.imshow(results[i],origin="lower",vmin=0,vmax=40000,cmap=cmap, aspect="auto")
        ax.set_title(titles[i], fontsize=8, pad=5)
    if i == 2:
        ax.set_xticks([0, 100, 200])
        ax.set_xticklabels([int(Ps[0] * 100), int(Ps[100] * 100), int(Ps[200] * 100)], fontsize=8)
        ax.set_xlabel("PrEP adoption (%))", fontsize=10)
    else:
        ax.set_xticks([0, 100, 200])
        ax.set_xticklabels([])
    ax.set_yticks([0, 100, 200])
    ax.set_yticklabels([int(Hs[0] * 100), int(Hs[100] * 100), int(Hs[200] * 100)], fontsize=8)
cbar = fig.colorbar(cax, ax=axs, shrink=0.99, pad=0.1)

# Add subpanel letters
for idx, ax in enumerate(axs):
    ax.text(0.15,0.95,chr(97 + idx),transform=ax.transAxes,fontsize=10,fontweight="bold",va="top",ha="right",color="white",)



# Second plot, signs of derivatives---------------------------------------------------------------------------------------------------

# set-up
#cmap = mcolors.ListedColormap(["black", 'white', '#E86C8A', '#1E88E5', '#056353']) 
cmap = mcolors.ListedColormap(["black", 'white', '#f6a582', '#d1e5f0', '#3a93c3']) 
bounds = [0, 1, 10, 100, 1000, 1001]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
right_grid = outer_grid[1].subgridspec(2, 3, wspace=0.2, hspace=0.2)
axes = []
for i in range(2):
    row_axes = []
    for j in range(3):
        ax = fig.add_subplot(right_grid[i, j])
        row_axes.append(ax)
    axes.append(row_axes)

# load data
results = [[], [], []]
for i, betaSTI in enumerate(beta_STI_values):
    res = []
    for lambdaP in lambda_P_values:
        with open( "../results/Nreal_Nobs_values_and_derivatives_lambdap%g_betaSTI%g_%s_dN%s.npy" %(lambdaP * 360, betaSTI, prevorinc, derivative),"rb") as f:
            Ps_plot = np.load(f)
            Hs_plot = np.load(f)
            dNobsdP_plot = np.load(f)
            #dNobsdH_plot = np.load(f)
            dNrealdP_plot = np.load(f)
            #dNrealdH_plot = np.load(f)
            Nobs_plot = np.load(f) * 100000
            Nreal_plot = np.load(f) * 100000
        x = dNrealdP_plot * dNobsdP_plot
        real_positive_obs_negative = np.where((dNrealdP_plot > 0) & (dNobsdP_plot < 0), 1, 0)
        real_negative_obs_positive = np.where((dNrealdP_plot < 0) & (dNobsdP_plot > 0), 10, 0)
        both_positive = np.where((dNrealdP_plot > 0) & (dNobsdP_plot > 0), 100, 0)
        both_negative = np.where((dNrealdP_plot < 0) & (dNobsdP_plot < 0), 1000, 0)
        variable_to_plot = real_positive_obs_negative + real_negative_obs_positive + both_positive + both_negative
        res.append(variable_to_plot)
    results[i] = res

global_vmin = -1e-7
global_vmax = 1e-7

# plotting
for row_idx, beta_STI in enumerate(beta_STI_values):
    for col_idx, (lambda_P, label) in enumerate(zip(lambda_P_values, lambda_P_labels)):
        ax = axes[row_idx][col_idx]
        cax = ax.imshow(results[row_idx][col_idx],origin="lower",cmap=cmap,norm=norm)
        ax.set_xticks([0, 100, 200])
        ax.set_xticklabels([int(Ps[0] * 100), int(Ps[100] * 100), int(Ps[200] * 100)], fontsize=8)
        ax.set_yticks([0, 100, 200])
        ax.set_yticklabels([int(Hs[0] * 100), int(Hs[50] * 100), int(Hs[100] * 100)], fontsize=8)

        if row_idx == 0:
            ax.set_title(f"{label}", fontsize=10, pad=5, color=colors_stability[col_idx])
        if col_idx == 0:
            ax.set_ylabel(f"{beta_STI_labels[row_idx]}", fontsize=9)


        if row_idx == 1:
            ax.set_xticks([0, 100, 200])
            ax.set_xticklabels([int(Ps[0] * 100), int(Ps[100] * 100), int(Ps[200] * 100)], fontsize=8)
        else:
            ax.set_xticks([0, 100, 200])
            ax.set_xticklabels([])

        if col_idx == 0:
            ax.set_yticks([0, 100, 200])
            ax.set_yticklabels([int(Hs[0] * 100), int(Hs[10] * 100), int(Hs[200] * 100)], fontsize=8)
        else:
            ax.set_yticks([0, 100, 200])
            ax.set_yticklabels([])

# Add subpanel letters
for idx, ax in enumerate([ax for row_axes in axes for ax in row_axes]):
    ax.text(0.15,0.95,chr(97 + 3 + idx),transform=ax.transAxes,fontsize=10,fontweight="bold",va="top",ha="right",color="white",)

# Add colorbar
tick_locs = [(bounds[i] + bounds[i+1]) / 2 for i in range(len(bounds)-1)]
#cbar = fig.colorbar(cax,ax=[ax for row_axes in axes for ax in row_axes],shrink=0.9,pad=0.01,ticks=tick_locs,)
#cbar.ax.set_yticklabels(["both don't change", "real increase,\nobserved decrease", "real decrease,\nobserved increase", "both increase", "both decrease"], fontsize=9)

fig.text(0.63, -0.05, "PrEP adoption (%)", ha="center", fontsize=10)
fig.text(0.03,0.5,"Risk awareness (%)",va="center",rotation="vertical",fontsize=10)

#plt.tight_layout()
#plt.show()

fig.savefig("../figures/final_figure_withinflux_DERIVATIVES_%s_dN%s.pdf" %(prevorinc, derivative), format="pdf", bbox_inches="tight")