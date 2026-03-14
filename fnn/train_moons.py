"""
train_moons.py
--------------
Trains a feedforward neural network (built from scratch with micrograd)
to classify the two-moons dataset.

Two crescent moon shapes sit on top of each other.
No straight line can separate them — but a neural network can.

Full story visualization (5 panels):
  1. Raw data — what the problem looks like
  2. Why a linear model fails
  3. What we want — a curved boundary
  4. Training the network — loss and accuracy
  5. What the network learned (yellow outline = misclassified)

Run:
    python train_moons.py
"""

# ============================================================
# TWO MOONS — Full story visualization
# ============================================================
# !pip install scikit-learn -q   ← uncomment and run once if needed

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_moons

from fnn import MLP, Value

# ── Reproducibility ───────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Dataset ───────────────────────────────────────────────
# make_moons generates two interleaved crescent shapes
# noise=0.15 adds a little randomness so it's not perfectly clean
X, y_raw = make_moons(n_samples=100, noise=0.15, random_state=42)

# convert labels 0/1 → -1/+1 to match tanh output range
y = [1.0 if label == 1 else -1.0 for label in y_raw]
xs = X.tolist()
ys = y

print(f"Dataset: {len(xs)} points")
print(f"Class +1 (top moon): {sum(1 for yi in ys if yi ==  1.0)} points")
print(f"Class -1 (bot moon): {sum(1 for yi in ys if yi == -1.0)} points")

# ── Colors ────────────────────────────────────────────────
C_POS  = '#58a6ff'   # blue  → class +1
C_NEG  = '#f78166'   # red   → class -1
C_TEXT = '#e6edf3'
C_GRID = '#21262d'
C_BG   = '#161b22'

X_arr    = np.array(X)
mask_pos = [yi ==  1.0 for yi in ys]
mask_neg = [yi == -1.0 for yi in ys]

# axis limits — used across multiple panels
x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3

# ── Figure: 5 panels ──────────────────────────────────────
fig = plt.figure(figsize=(22, 8), facecolor='#0d1117')
gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.38)

# ─────────────────────────────────────────────────────────
# PANEL 1: Raw data — what does the problem look like?
# ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(C_BG)

ax1.scatter(X_arr[mask_pos, 0], X_arr[mask_pos, 1],
            c=C_POS, s=50, edgecolors='white', linewidths=0.5,
            label='Class +1 (top moon)', zorder=5)
ax1.scatter(X_arr[mask_neg, 0], X_arr[mask_neg, 1],
            c=C_NEG, s=50, edgecolors='white', linewidths=0.5,
            marker='s', label='Class -1 (bot moon)', zorder=5)

ax1.set_title('Step 1: The Raw Data\n(two moons, 100 points)',
              color=C_TEXT, fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel('x₁', color='#8b949e')
ax1.set_ylabel('x₂', color='#8b949e')
ax1.tick_params(colors='#8b949e')
ax1.spines[:].set_color('#30363d')
ax1.grid(True, color=C_GRID, linewidth=0.8, alpha=0.5)
ax1.legend(facecolor='#21262d', edgecolor='#30363d',
           labelcolor=C_TEXT, fontsize=8, loc='upper right')

# ─────────────────────────────────────────────────────────
# PANEL 2: Why a linear model fails
# ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(C_BG)

ax2.scatter(X_arr[mask_pos, 0], X_arr[mask_pos, 1],
            c=C_POS, s=50, edgecolors='white', linewidths=0.5, zorder=5)
ax2.scatter(X_arr[mask_neg, 0], X_arr[mask_neg, 1],
            c=C_NEG, s=50, edgecolors='white', linewidths=0.5,
            marker='s', zorder=5)

# draw a straight line — the best a linear model can do
x_line = np.linspace(-1.5, 2.5, 100)
ax2.plot(x_line, 0.3 * x_line + 0.1,
         color='#f0e68c', linewidth=2.5, linestyle='--',
         label='Best linear attempt', zorder=4)

ax2.set_title('Step 2: Linear Model Fails\n(no straight line works)',
              color=C_TEXT, fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel('x₁', color='#8b949e')
ax2.set_ylabel('x₂', color='#8b949e')
ax2.tick_params(colors='#8b949e')
ax2.spines[:].set_color('#30363d')
ax2.grid(True, color=C_GRID, linewidth=0.8, alpha=0.5)
ax2.legend(facecolor='#21262d', edgecolor='#30363d',
           labelcolor=C_TEXT, fontsize=8)

# ─────────────────────────────────────────────────────────
# PANEL 3: What we WANT — a curved boundary
# ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor(C_BG)

# approximate ideal boundary using a sine curve
xx_d, yy_d = np.meshgrid(np.linspace(x_min, x_max, 80),
                          np.linspace(y_min, y_max, 80))
Z_ideal = np.sin(xx_d * 2.2) * 0.55 - yy_d

ax3.contourf(xx_d, yy_d, Z_ideal, levels=[-10, 0, 10],
             colors=[C_NEG, C_POS], alpha=0.2)
ax3.contour(xx_d, yy_d, Z_ideal, levels=[0],
            colors='white', linewidths=2.5, linestyles='--')

ax3.scatter(X_arr[mask_pos, 0], X_arr[mask_pos, 1],
            c=C_POS, s=50, edgecolors='white', linewidths=0.5, zorder=5)
ax3.scatter(X_arr[mask_neg, 0], X_arr[mask_neg, 1],
            c=C_NEG, s=50, edgecolors='white', linewidths=0.5,
            marker='s', zorder=5)

ax3.set_title('Step 3: What We Want\n(a curved boundary)',
              color=C_TEXT, fontsize=11, fontweight='bold', pad=10)
ax3.set_xlabel('x₁', color='#8b949e')
ax3.set_ylabel('x₂', color='#8b949e')
ax3.tick_params(colors='#8b949e')
ax3.spines[:].set_color('#30363d')
ax3.grid(True, color=C_GRID, linewidth=0.8, alpha=0.5)

# ─────────────────────────────────────────────────────────
# TRAIN THE NETWORK
# 2 inputs (x,y coords) → 16 neurons → 16 neurons → 1 output
# wider than XOR because moons is a harder, real-shaped problem
# ─────────────────────────────────────────────────────────
n = MLP(nin=2, nouts=[16, 16, 1])
print(f"\nModel: MLP(2 → 16 → 16 → 1)")
print(f"Total parameters: {len(n.parameters())}")
print("\nTraining...")

losses      = []
accuracies  = []

for step in range(100):

    # forward pass
    ypred = [n(x) for x in xs]

    # hinge loss: max(0, 1 - y*pred)
    # better than MSE for classification — if you're already correct
    # AND confident, your loss contribution is exactly 0
    data_loss = sum(
        (1 + -yi * yout) if (1 + -yi * yout.data) > 0 else Value(0.0)
        for yi, yout in zip(ys, ypred)
    ) * (1.0 / len(ys))

    # L2 regularization: tiny penalty on large weights
    # stops any one weight from dominating → prevents overfitting
    reg_loss = sum(p * p for p in n.parameters()) * 1e-4

    loss = data_loss + reg_loss

    # zero gradients → backward → update
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
    for p in n.parameters():
        p.data -= 0.05 * p.grad

    # track metrics
    predicted_labels = [1.0 if yout.data > 0 else -1.0 for yout in ypred]
    acc = sum(pl == gt for pl, gt in zip(predicted_labels, ys)) / len(ys)
    losses.append(loss.data)
    accuracies.append(acc * 100)

    if step % 10 == 0 or step == 99:
        print(f"  step {step:3d} | loss = {loss.data:.4f} | accuracy = {acc*100:.0f}%")

# ─────────────────────────────────────────────────────────
# PANEL 4: Training curves — loss and accuracy together
# ─────────────────────────────────────────────────────────
ax4     = fig.add_subplot(gs[3])
ax4_r   = ax4.twinx()   # second y-axis on the right for accuracy
ax4.set_facecolor(C_BG)
ax4_r.set_facecolor(C_BG)

ax4.plot(losses, color=C_POS, linewidth=2.5, label='Loss')
ax4.fill_between(range(len(losses)), losses, alpha=0.1, color=C_POS)

ax4_r.plot(accuracies, color='#3fb950', linewidth=2,
           linestyle='-.', label='Accuracy %')
ax4_r.set_ylim(0, 110)
ax4_r.tick_params(colors='#8b949e')
ax4_r.spines[:].set_color('#30363d')
ax4_r.set_ylabel('Accuracy (%)', color='#3fb950', fontsize=9)

ax4.set_title('Step 4: Training the Network\n(loss ↓  accuracy ↑)',
              color=C_TEXT, fontsize=11, fontweight='bold', pad=10)
ax4.set_xlabel('Training Step', color='#8b949e')
ax4.set_ylabel('Loss', color='#8b949e')
ax4.tick_params(colors='#8b949e')
ax4.spines[:].set_color('#30363d')
ax4.grid(True, color=C_GRID, linewidth=0.8, alpha=0.5)

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_r.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2,
           facecolor='#21262d', edgecolor='#30363d',
           labelcolor=C_TEXT, fontsize=8)

# ─────────────────────────────────────────────────────────
# PANEL 5: What the network actually learned
# The colored background = network's decision regions
# White dashed line = learned decision boundary
# Yellow outline = misclassified point
# ─────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[4])
ax5.set_facecolor(C_BG)

# run every point of a fine grid through the trained network
# this lets us color the entire space based on what the network predicts
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 60),
                      np.linspace(y_min, y_max, 60))
grid_points = np.c_[xx.ravel(), yy.ravel()].tolist()
Z = np.array([n(pt).data for pt in grid_points]).reshape(xx.shape)

# color the decision regions
ax5.contourf(xx, yy, Z, levels=50, cmap=plt.cm.RdYlBu, alpha=0.55)
# draw the decision boundary where network output = 0
ax5.contour(xx, yy, Z, levels=[0],
            colors='white', linewidths=2.5, linestyles='--')

# plot each data point — yellow outline means the network got it wrong
final_preds  = [1.0 if n(x).data > 0 else -1.0 for x in xs]
correct_mask = [p == g for p, g in zip(final_preds, ys)]
final_acc    = sum(correct_mask) / len(correct_mask)

for xi, yi_true, correct in zip(xs, ys, correct_mask):
    color  = C_POS if yi_true ==  1.0 else C_NEG
    marker = 'o'   if yi_true ==  1.0 else 's'
    edge   = 'white'   if correct else '#ffff00'  # yellow = wrong
    lw     = 0.5       if correct else 2.0
    ax5.scatter(xi[0], xi[1], c=color, s=55,
                edgecolors=edge, linewidths=lw,
                marker=marker, zorder=5)

ax5.set_title(f'Step 5: What the Network Learned\n'
              f'accuracy: {final_acc*100:.0f}%  |  yellow outline = wrong',
              color=C_TEXT, fontsize=11, fontweight='bold', pad=10)
ax5.set_xlabel('x₁', color='#8b949e')
ax5.set_ylabel('x₂', color='#8b949e')
ax5.tick_params(colors='#8b949e')
ax5.spines[:].set_color('#30363d')
ax5.grid(True, color=C_GRID, linewidth=0.8, alpha=0.5)

# ── Final title ────────────────────────────────────────────
plt.suptitle('Neural Network from Scratch — Two Moons Classification',
             color=C_TEXT, fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('moons_results.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("Plot saved → moons_results.png")
