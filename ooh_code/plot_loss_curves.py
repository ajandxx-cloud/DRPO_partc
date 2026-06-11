"""Plot publication-quality Huber loss curves for DSPO and DRPO."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- DSPO loss data extracted from logfile ---
dspo_loss = np.array([
    2056.78, 1447.64, 1208.52, 1029.14, 898.08, 821.20, 756.75, 696.22,
    650.89, 610.91, 427.10, 365.73, 312.36, 280.96, 260.29, 232.40,
    209.56, 195.28, 179.70, 167.11, 156.93, 145.50, 135.10, 125.89,
    116.95, 108.20, 100.44, 93.70, 86.10, 79.31, 73.16, 67.08,
    62.33, 58.21, 54.31, 51.05, 48.23, 45.38, 42.94, 39.88,
    37.16, 34.88, 32.76, 30.54, 28.87, 27.13, 25.37, 23.84,
    22.48, 21.53
])

# --- DRPO loss data extracted from logfile ---
drpo_loss = np.array([
    1716.88, 1390.91, 1360.53, 1199.32, 1090.06, 1000.13, 902.20, 835.54,
    767.08, 717.12, 565.36, 480.43, 368.32, 314.89, 267.54, 228.52,
    214.41, 191.79, 184.12, 168.19, 160.03, 147.65, 139.62, 129.65,
    119.38, 109.88, 98.62, 90.77, 82.03, 76.43, 69.96, 65.37,
    59.97, 55.89, 51.64, 48.64, 46.04, 42.85, 40.51, 38.17,
    35.48, 33.48, 31.42, 29.62, 28.38, 27.10, 25.75, 24.80,
    23.66, 22.80
])

epochs = np.arange(len(dspo_loss))

# --- Style setup -------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.6,
    'axes.linewidth': 0.8,
})

output_dir = r'c:\Users\39583\Desktop\4_Publication\DRT\figures'
PREFIX = 'loss_curves_v2'

# --- Color palette -----------------------------------------------------------
C_DSPO = '#2c7bb6'      # muted blue
C_DRPO = '#d7191c'  # muted red

# =============================================================================
# Figure 1: Linear scale, side by side (two subplots)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# -- DSPO --
ax1.plot(epochs, dspo_loss, color=C_DSPO, linewidth=1.8)
ax1.scatter(0, dspo_loss[0], color=C_DSPO, s=35, zorder=5)
ax1.scatter(len(dspo_loss)-1, dspo_loss[-1], color=C_DSPO, s=35, zorder=5,
           marker='D')
ax1.annotate(f'{dspo_loss[0]:.0f}', xy=(0, dspo_loss[0]),
             xytext=(4, dspo_loss[0] + 120), fontsize=9, color=C_DSPO,
             arrowprops=dict(arrowstyle='->', color=C_DSPO, lw=0.8))
ax1.annotate(f'{dspo_loss[-1]:.1f}', xy=(len(dspo_loss)-1, dspo_loss[-1]),
             xytext=(len(dspo_loss)-8, dspo_loss[-1] + 180), fontsize=9, color=C_DSPO,
             arrowprops=dict(arrowstyle='->', color=C_DSPO, lw=0.8))
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Huber Loss')
ax1.set_title('DSPO')
ax1.set_xlim(-1, len(dspo_loss))
ax1.grid(True, alpha=0.3, lw=0.5)
ax1.set_ylim(bottom=0)

# -- DRPO --
ax2.plot(epochs, drpo_loss, color=C_DRPO, linewidth=1.8)
ax2.scatter(0, drpo_loss[0], color=C_DRPO, s=35, zorder=5)
ax2.scatter(len(drpo_loss)-1, drpo_loss[-1], color=C_DRPO, s=35, zorder=5,
           marker='D')
ax2.annotate(f'{drpo_loss[0]:.0f}', xy=(0, drpo_loss[0]),
             xytext=(4, drpo_loss[0] + 120), fontsize=9, color=C_DRPO,
             arrowprops=dict(arrowstyle='->', color=C_DRPO, lw=0.8))
ax2.annotate(f'{drpo_loss[-1]:.1f}', xy=(len(drpo_loss)-1, drpo_loss[-1]),
             xytext=(len(drpo_loss)-8, drpo_loss[-1] + 150), fontsize=9, color=C_DRPO,
             arrowprops=dict(arrowstyle='->', color=C_DRPO, lw=0.8))
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Huber Loss')
ax2.set_title('DRPO')
ax2.set_xlim(-1, len(drpo_loss))
ax2.grid(True, alpha=0.3, lw=0.5)
ax2.set_ylim(bottom=0)

fig.suptitle('Supervised Pre-Training Loss Convergence', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(f'{output_dir}/{PREFIX}_side_by_side.pdf', format='pdf')
fig.savefig(f'{output_dir}/{PREFIX}_side_by_side.png', format='png')
plt.close(fig)
print('Saved: loss_curves_side_by_side.pdf / .png')


# =============================================================================
# Figure 2: Overlaid comparison (both curves on single axes)
# =============================================================================
fig, ax = plt.subplots(figsize=(6, 4.5))

ax.plot(epochs, dspo_loss, color=C_DSPO, linewidth=1.8, label='DSPO')
ax.plot(epochs, drpo_loss, color=C_DRPO, linewidth=1.8, label='DRPO')

ax.scatter(0, dspo_loss[0], color=C_DSPO, s=30, zorder=5)
ax.scatter(0, drpo_loss[0], color=C_DRPO, s=30, zorder=5)
ax.scatter(len(dspo_loss)-1, dspo_loss[-1], color=C_DSPO, s=40, zorder=5, marker='D')
ax.scatter(len(drpo_loss)-1, drpo_loss[-1], color=C_DRPO, s=40, zorder=5, marker='D')

ax.set_xlabel('Epoch')
ax.set_ylabel('Huber Loss')
ax.set_xlim(-1, len(dspo_loss))
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3, lw=0.5)
ax.legend(frameon=True, fancybox=False, edgecolor='gray', facecolor='white',
          loc='upper right')

fig.tight_layout()
fig.savefig(f'{output_dir}/{PREFIX}_overlaid.pdf', format='pdf')
fig.savefig(f'{output_dir}/{PREFIX}_overlaid.png', format='png')
plt.close(fig)
print('Saved: loss_curves_overlaid.pdf / .png')


# =============================================================================
# Figure 3: Log-scale version (better reveals convergence behavior)
# =============================================================================
fig, ax = plt.subplots(figsize=(6, 4.5))

ax.semilogy(epochs, dspo_loss, color=C_DSPO, linewidth=1.8, label='DSPO')
ax.semilogy(epochs, drpo_loss, color=C_DRPO, linewidth=1.8, label='DRPO')

ax.scatter(0, dspo_loss[0], color=C_DSPO, s=30, zorder=5)
ax.scatter(0, drpo_loss[0], color=C_DRPO, s=30, zorder=5)
ax.scatter(len(dspo_loss)-1, dspo_loss[-1], color=C_DSPO, s=40, zorder=5, marker='D')
ax.scatter(len(drpo_loss)-1, drpo_loss[-1], color=C_DRPO, s=40, zorder=5, marker='D')

ax.set_xlabel('Epoch')
ax.set_ylabel('Huber Loss (log scale)')
ax.set_xlim(-1, len(dspo_loss))
ax.grid(True, alpha=0.3, lw=0.5, which='both')
ax.legend(frameon=True, fancybox=False, edgecolor='gray', facecolor='white',
          loc='upper right')

fig.tight_layout()
fig.savefig(f'{output_dir}/{PREFIX}_log.pdf', format='pdf')
fig.savefig(f'{output_dir}/{PREFIX}_log.png', format='png')
plt.close(fig)
print('Saved: loss_curves_log.pdf / .png')

# --- Summary stats -----------------------------------------------------------
print(f'\nDSPO:   epoch 0 = {dspo_loss[0]:.1f}, epoch 49 = {dspo_loss[-1]:.1f}, '
      f'reduction = {(1 - dspo_loss[-1]/dspo_loss[0])*100:.1f}%')
print(f'DRPO:   epoch 0 = {drpo_loss[0]:.1f}, epoch 49 = {drpo_loss[-1]:.1f}, '
      f'reduction = {(1 - drpo_loss[-1]/drpo_loss[0])*100:.1f}%')
print('Done.')
