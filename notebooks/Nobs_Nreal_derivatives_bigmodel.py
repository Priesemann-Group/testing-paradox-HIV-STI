import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import sys
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from tapm import bigmodel_STI
import icomo

sys.path.append("../src/")
from tapm import utils

jax.config.update("jax_enable_x64", True)


lambdaPs = [1/360, 2/360, 4/360] 
betaSTIs = [0.0016*5, 0.0016*7]  

# here you have to choose what you want
onlyplots = False # if True, only plots are generated, if False, first data is generated and then plots
derivative = "dP" # dP: derivative with respect to PrEP adoption, dH: derivative with respect to risk awareness

which_c = 9
which_xi = 0

sets_of_c = jnp.array([
    [31.0,  40.0,   60.0,  203.0],
    [29.0,  38.0,   73.0,  203.0],
    [14.0,  24.0,  167.0,  203.0], #2
    [15.0,  40.0,  120.0,  203.0],
    [10.0,  38.0,  141.3, 203.0],
    [2.0, 80.0, 53.3, 200.0],
    [2.0, 63.5, 100.0, 200.0], #6
    [75, 39, 18.5, 2], #7
    [10.0, 50.0, 120.0, 190.0], #8
    [50.0, 50.0, 50.0, 50.0] #9
])
sets_of_xi = jnp.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.8, 0.6, 0.4, 0.2], #1
    [0.2, 0.4, 0.6, 0.8] #2
])



Ps = np.linspace(0, 1, 201)
Hs = np.linspace(0, 0.2, 201)

# helper stuff, don't change
if derivative == "dP":
    argnums = 0
else:
    argnums = 1
args = bigmodel_STI.args
y0 = bigmodel_STI.y0

# functions to calculate Nobs
def lambdas(P, H): # symptomatic testing
    return args["lambda_0"] + lambdaa(P, H)

def lambdaa(P, H): # asymptomatic testing
    return lambdaH(P, H) * (1 - P) + lambdaP * P

def lambdaH(P, H): # risk-related testing
    return args["c"] * (1 - m(P, H)) * args["beta_HIV"] * H

def m(P, H): # mitigation
    return args["m_min"] + (args["m_max"] - args["m_min"]) * (1 - jnp.exp(-H / args["H_thres"]))

# calculates Nreal for given H, P, lambdaP and betaSTI
def calc_Nreal(H, P, lambdaP, betaSTI):

    # Set the parameters
    args_mod = args.copy()
    args_mod["H"] = H
    args_mod["P"] = P
    args_mod["lambda_P"] = lambdaP
    args_mod["beta_STI"] = betaSTI
    args_mod["c"] = sets_of_c[which_c]
    args_mod["xi"] = sets_of_xi[which_xi]  # partial mitigation for people on PrEP

    # run the model fro 80 years (long time to get to steady state)
    output = icomo.diffeqsolve(args = args_mod, ODE = bigmodel_STI.main_model, y0 = y0, ts_out = np.linspace(0, 365*80, 365*80+1), max_steps=365*80+1)

    # Get the final state of the system (equilibrium value)
    y1 = {key: value[-1] for key, value in output.ys.items()}
    #check convergence by comparing last value to fifth last value
    y2 = {key: value[-5] for key, value in output.ys.items()}
    for key in y1:
        if not np.allclose(y1[key], y2[key], rtol=0, atol=1e-8):
            print(f"Convergence failed: {key}: (diff = {abs(y1[key] - y2[key])})")

    return jnp.sum(y1["Ia_STI"] + y1["Is_STI"])

# calculates Nobs for given H, P, lambdaP and betaSTI
def calc_Nobs(H, P, lambdaP, betaSTI):

    # Set the parameters
    args_mod = args.copy()
    args_mod["H"] = H
    args_mod["P"] = P
    args_mod["lambda_P"] = lambdaP
    args_mod["beta_STI"] = betaSTI
    args_mod["c"] = sets_of_c[which_c]

    # run the model fro 80 years (long time to get to steady state)
    output = icomo.diffeqsolve(args = args_mod, ODE = bigmodel_STI.main_model, y0 = y0, ts_out = np.linspace(0, 365*80, 365*80+1), max_steps=365*80+1)

    # Get the final state of the system (equilibrium value)
    y1 = {key: value[-1] for key, value in output.ys.items()}
    #check convergence by comparing last value to fifth last value
    y2 = {key: value[-5] for key, value in output.ys.items()}
    for key in y1:
        if not np.allclose(y1[key], y2[key], rtol=0, atol=1e-8):
            print(f"Convergence failed: {key}: (diff = {abs(y1[key] - y2[key])})")

    return np.sum(lambdas(P,H) * y1["Is_STI"] + lambdaa(P,H) * y1["Ia_STI"])

if not onlyplots:
    for lambdaP in lambdaPs:
        for betaSTI in betaSTIs:

            Nreal_res = np.zeros((len(Ps), len(Hs)))
            Nobs_res = np.zeros((len(Ps), len(Hs)))
            dNrealdP = np.zeros((len(Ps)-1, len(Hs)-1))
            dNobsdP = np.zeros((len(Ps)-1, len(Hs)-1))
            dNrealdH = np.zeros((len(Ps)-1, len(Hs)-1))
            dNobsdH = np.zeros((len(Ps)-1, len(Hs)-1))

            # calculate Nreal and Nobs
            for i,H in enumerate(Hs):
                for j,P in enumerate(Ps):
                    Nreal_res[i,j] = calc_Nreal(H, P, lambdaP, betaSTI)
                    Nobs_res[i,j] = calc_Nobs(H, P, lambdaP, betaSTI)

            #calculate sign of derivative (compare to previous value and see if it is bigger or smaller)
            for i,H in enumerate(Hs[:-1]):
                for j,P in enumerate(Ps[:-1]):
                    dNrealdP[i,j] = np.sign(Nreal_res[i,j+1] - Nreal_res[i,j]) # if Nreal increases, derivative is positive
                    dNobsdP[i,j] = np.sign(Nobs_res[i,j+1] - Nobs_res[i,j]) # if Nobs increases, derivative is positive
                    dNrealdH[i,j] = np.sign(Nreal_res[i+1,j] - Nreal_res[i,j]) # if Nreal increases, derivative is positive
                    dNobsdH[i,j] = np.sign(Nobs_res[i+1,j] - Nobs_res[i,j]) # if Nobs increases, derivative is positive

            # save stuff as npy files

            with open(
                "../results/Nreal_Nobs_bigmodel_lambdap%g_betaSTI%g_dN%s_c%s_xi%s.npy" %(lambdaP * 360, betaSTI, derivative, which_c, which_xi),"wb",) as f:
                np.save(f, Ps)
                np.save(f, Hs)
                np.save(f, dNrealdP)
                np.save(f, dNobsdP)
                np.save(f, dNrealdH)
                np.save(f, dNobsdH)


# Plotting----------------------------------------------------------------------------------------------------------------------------

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



# Second plot, signs of derivatives---------------------------------------------------------------------------------------------------

# set-up
# nothing, Nreal increases Nobs decreases, Nreal decreases Nobs increases, both increase, both decrease
cmap = mcolors.ListedColormap(["black", '#f6a582', '#f6a582', '#d1e5f0', '#3a93c3']) 
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
        # load data
        with open("../results/Nreal_Nobs_bigmodel_lambdap%g_betaSTI%g_dN%s_c%s_xi%s.npy" %(lambdaP * 360, betaSTI, derivative, which_c, which_xi),"rb") as f:
            Ps_plot = np.load(f)
            Hs_plot = np.load(f)
            dNrealdP_plot = np.load(f)
            dNobsdP_plot = np.load(f)
            dNrealdH_plot = np.load(f)
            dNobsdH_plot = np.load(f)

        if derivative == "dP":
            real_positive_obs_negative = np.where((dNrealdP_plot > 0) & (dNobsdP_plot < 0), 1, 0)
            real_negative_obs_positive = np.where((dNrealdP_plot < 0) & (dNobsdP_plot > 0), 10, 0)
            both_positive = np.where((dNrealdP_plot > 0) & (dNobsdP_plot > 0), 100, 0)
            both_negative = np.where((dNrealdP_plot < 0) & (dNobsdP_plot < 0), 1000, 0)
        else:
            real_positive_obs_negative = np.where((dNrealdH_plot > 0) & (dNobsdH_plot < 0), 1, 0)
            real_negative_obs_positive = np.where((dNrealdH_plot < 0) & (dNobsdH_plot > 0), 10, 0)
            both_positive = np.where((dNrealdH_plot > 0) & (dNobsdH_plot > 0), 100, 0)
            both_negative = np.where((dNrealdH_plot < 0) & (dNobsdH_plot < 0), 1000, 0)
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

fig.savefig("../figures/final_figure_withinflux_bigmodel_DERIVATIVES_dN%s_c%s_xi%s.pdf" %(derivative, which_c, which_xi), format="pdf", bbox_inches="tight")