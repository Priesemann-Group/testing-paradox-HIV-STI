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

args, y0 = utils.read_params(filename="model_params_HIVandSTI.txt")

derivative = "dP"
lambdaPs = [1/360, 2/360, 4/360] 
betaSTIs = [0.0016*5]  # only one beta

metrics = [r"$N$", r"$N_{\text{obs}}$", r"$N - N_{\text{obs}}$"]
row_vlims = [(0, 70000), (0, 110), (0, 70000)]  # fixed limits per row

# --- figure setup ---
fig, axs = plt.subplots(len(metrics), len(lambdaPs), figsize=(7.1, 5.5))
plt.subplots_adjust(left=0.18, right=0.85, top=0.95, bottom=0.14, hspace=0.3, wspace=0.25)

# --- load data and plot ---
for col, lambdaP in enumerate(lambdaPs):
    betaSTI = betaSTIs[0]  # only one beta
    with open(
        "../results/Nreal_Nobs_values_and_derivatives_lambdap%g_betaSTI%g_%s_dN%s_sigma%g.npy"
        % (lambdaP*360, betaSTI, "prevalence", derivative, args["Sigma"]),
        "rb"
    ) as f:
        Ps = np.load(f)
        Hs = np.load(f)
        _ = np.load(f)  # dNobsdP
        _ = np.load(f)  # dNrealdP
        Nobs = np.load(f) * 100000
        Nreal = np.load(f) * 100000

    mats = [Nreal, Nobs, Nreal - Nobs]

    for row, mat in enumerate(mats):
        vmin, vmax = row_vlims[row]
        ax = axs[row, col] if len(lambdaPs) > 1 else axs[row]
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)

# --- format axes ---
for row in range(len(metrics)):
    for col in range(len(lambdaPs)):
        ax = axs[row, col] if len(lambdaPs) > 1 else axs[row]
        # top titles = lambdaP
        if row == 0:
            ax.set_title(r"$\lambda_P = %g$" % (lambdaPs[col]*360))

        # x ticks only on bottom row
        if row == len(metrics)-1:
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

    # --- add row metric label further left and rotated ---
    ax_metric = axs[row,0] if len(lambdaPs) > 1 else axs[row]
    ax_metric.annotate(
        metrics[row],
        xy=(-0.6, 0.5), xycoords="axes fraction",
        ha="center", va="center", fontsize=9,
        rotation=90
    )

# --- vertical colorbars for each row ---
for row, (vmin, vmax) in enumerate(row_vlims):
    pos_left = axs[row, -1].get_position() if len(lambdaPs) > 1 else axs[row].get_position()
    cax = fig.add_axes([
        pos_left.x1 + 0.01,  # right of last column
        pos_left.y0,
        0.02,                # width
        pos_left.height      # height same as row
    ])
    cbar = fig.colorbar(axs[row,0].images[0] if len(lambdaPs) > 1 else axs[row].images[0],
                        cax=cax, orientation="vertical")
    # center label on colorbar, adjust x-position moderately
    cbar.ax.yaxis.set_label_coords(4.55, 0.5)
    cbar.set_label("Population (per 100,000)")
    cbar.set_ticks([vmin, (vmin+vmax)//2, vmax])

# --- save ---
plt.savefig("../figures/Nreal_Nobs_prevalence_difference_sigma%g.pdf" % args["Sigma"],
            bbox_inches='tight')
plt.close()
