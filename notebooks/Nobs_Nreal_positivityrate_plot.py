import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../src/")
from tapm import utils

# ---- style tweaks ----
plt.rc('font', size=10)
plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=8)
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('figure', titlesize=10)

# --- parameters ---
args, y0 = utils.read_params(filename="model_params_HIVandSTI.txt")

derivative = "dP"
lambdaPs = [1/360, 2/360, 4/360]       # columns
betaSTIs = [0.0016*5, 0.0016*7]        # rows, example values
nrows = len(betaSTIs)
ncols = len(lambdaPs)

# --- figure setup ---
fig, axs = plt.subplots(nrows, ncols, figsize=(7.1, 5.5))
plt.subplots_adjust(left=0.18, right=0.88, top=0.95, bottom=0.14, hspace=0.3, wspace=0.25)

# --- first pass: find global vmin/vmax for colorbar ---
all_rates = []

for row, betaSTI in enumerate(betaSTIs):
    for col, lambdaP in enumerate(lambdaPs):
        with open(
            "../results/Nreal_Nobs_values_and_derivatives_lambdap%g_betaSTI%g_%s_dN%s_sigma%g.npy"
        % (lambdaP*360, betaSTI, "prevalence", derivative, args["Sigma"]),
            "rb"
        ) as f:
            Ps = np.load(f)
            Hs = np.load(f)
            dNobsdP = np.load(f)
            dNrealdP = np.load(f)
            Nobs = np.load(f)
            Nreal = np.load(f)
            S = np.load(f)
            Ia = np.load(f)
            Is = np.load(f)
            lambdaa_vals = np.load(f)
            lambdas_vals = np.load(f)

        T = 1 - Ia - Is - S
        positivity = (Ia*lambdaa_vals + Is*lambdas_vals) / (T*lambdaP + Ia*lambdaa_vals + Is*lambdas_vals + S*lambdaa_vals)
        all_rates.append(positivity)

vmin = 0
vmax = 1

# --- second pass: plot ---
for row, betaSTI in enumerate(betaSTIs):
    for col, lambdaP in enumerate(lambdaPs):
        with open("../results/Nreal_Nobs_values_and_derivatives_lambdap%g_betaSTI%g_%s_dN%s_sigma%g.npy" % (lambdaP*360, betaSTI, "prevalence", derivative, args["Sigma"]), "rb") as f:
            Ps = np.load(f)
            Hs = np.load(f)
            _ = np.load(f)  # dNobsdP
            _ = np.load(f)  # dNrealdP
            _ = np.load(f)  # Nobs
            _ = np.load(f)  # Nreal
            S = np.load(f)
            Ia = np.load(f)
            Is = np.load(f)
            lambdaa_vals = np.load(f)
            lambdas_vals = np.load(f)

        T = 1 - Ia - Is - S
        positivity = (Ia*lambdaa_vals + Is*lambdas_vals) / (T*lambdaP + Ia*lambdaa_vals + Is*lambdas_vals + S*lambdaa_vals)

        ax = axs[row, col] if nrows > 1 else axs[col]
        im = ax.imshow(positivity, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)

# --- format axes ---
for row in range(nrows):
    for col in range(ncols):
        ax = axs[row, col] if nrows > 1 else axs[col]
        # top titles = lambdaP
        if row == 0:
            ax.set_title(r"$\lambda_P = %g$" % (lambdaPs[col]*360))

        # x ticks only on bottom row
        if row == nrows-1:
            ax.set_xticks([0, 100, 200])
            ax.set_xticklabels([int(Ps[0]*100), int(Ps[100]*100), int(Ps[200]*100)])
            ax.set_xlabel(r"PrEP uptake (%)")
        else:
            ax.set_xticks([])

        # y ticks only first column
        if col == 0:
            ax.set_yticks([0, 100, 200])
            ax.set_yticklabels([int(Hs[0]*100), int(Hs[50]*100), int(Hs[100]*100)])
            ax.set_ylabel(r"Risk awareness $H$ (%)", labelpad=8)
        else:
            ax.set_yticks([])

# --- add row beta_STI labels further left and rotated ---
for row, betaSTI in enumerate(betaSTIs):
    ax_metric = axs[row,0] if nrows > 1 else axs[row]
    ax_metric.annotate(
        r"$\beta_{\mathrm{STI}} = %g$" % betaSTI,
        xy=(-0.6, 0.5), xycoords="axes fraction",
        ha="center", va="center", fontsize=9,
        rotation=90
    )

# --- single colorbar for all plots ---
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # x0, y0, width, height
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
cbar.set_label("Positivity rate")
cbar.set_ticks([0,0.5,1])

# --- save figure ---
plt.savefig("../figures/positivity_rate_sigma%g.pdf" % args["Sigma"],
            bbox_inches='tight')
plt.close()
