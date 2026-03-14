"""
train_moons.py
--------------
Trains a feedforward neural network (built from scratch with micrograd)
to classify the two-moons dataset.

Two crescent moon shapes sit on top of each other.
No straight line can separate them — but a neural network can.

The final plot shows:
  1. Training loss over time
  2. The decision boundary the network learned
  3. Final predictions on every training point

Run:
    python train_moons.py
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_moons

from fnn import MLP, Value

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Dataset ───────────────────────────────────────────────────────────────────
# make_moons generates two interleaved crescent shapes
# noise adds a little randomness so it's not perfectly clean
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# convert labels from 0/1 → -1/+1 to match tanh output range
y = [1.0 if label == 1 else -1.0 for label in y]

# convert to plain python lists so micrograd can handle them
xs = X.tolist()
ys = y

print(f"Dataset: {len(xs)} points, 2 classes")
print(f"Class +1 (top moon): {sum(1 for yi in ys if yi == 1.0)} points")
print(f"Class -1 (bot moon): {sum(1 for yi in ys if yi == -1.0)} points")

# ── Model ─────────────────────────────────────────────────────────────────────
# 2 inputs (x, y coordinates) → 16 neurons → 16 neurons → 1 output
# deeper/wider than XOR because moons is a harder problem
n = MLP(nin=2, nouts=[16, 16, 1])
print(f"\nModel: MLP(2 → 16 → 16 → 1)")
print(f"Total trainable parameters: {len(n.parameters())}")

# ── Hyperparameters ───────────────────────────────────────────────────────────
STEPS         = 100
LEARNING_RATE = 0.05

# ── Training loop ─────────────────────────────────────────────────────────────
losses = []
accuracies = []

print("\nTraining...")
for step in range(STEPS):

    # 1. forward pass — run every point through the network
    ypred = [n(x) for x in xs]

    # 2. hinge loss — works better than MSE for classification
    #    max(0, 1 - y*pred) penalizes wrong predictions more aggressively
    #    if prediction is correct AND confident → loss contribution is 0
    #    if prediction is wrong → loss grows linearly
    data_loss = sum(
        (1 + -yi * yout) if (1 + -yi * yout.data) > 0 else Value(0.0)
        for yi, yout in zip(ys, ypred)
    ) * (1.0 / len(ys))

    # 3. L2 regularization — gently pushes weights toward 0
    #    prevents any single weight from becoming too dominant (overfitting)
    #    alpha controls how strong the regularization is
    alpha = 1e-4
    reg_loss = sum(p * p for p in n.parameters()) * alpha
    loss = data_loss + reg_loss

    # 4. zero gradients before backward
    for p in n.parameters():
        p.grad = 0.0

    # 5. backward pass — compute all gradients
    loss.backward()

    # 6. gradient descent — update weights
    for p in n.parameters():
        p.data -= LEARNING_RATE * p.grad

    # 7. track accuracy — how many did we get right?
    predicted_labels = [1.0 if yout.data > 0 else -1.0 for yout in ypred]
    acc = sum(pl == gt for pl, gt in zip(predicted_labels, ys)) / len(ys)

    losses.append(loss.data)
    accuracies.append(acc * 100)

    if step % 10 == 0 or step == STEPS - 1:
        print(f"step {step:3d} | loss = {loss.data:.4f} | accuracy = {acc*100:.0f}%")

# ── Final accuracy ────────────────────────────────────────────────────────────
final_preds = [1.0 if n(x).data > 0 else -1.0 for x in xs]
final_acc   = sum(p == g for p, g in zip(final_preds, ys)) / len(ys)
print(f"\nFinal accuracy: {final_acc*100:.0f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6), facecolor='#0d1117')
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# shared colors
C_POS  = '#58a6ff'   # blue  → class +1
C_NEG  = '#f78166'   # red   → class -1
C_TEXT = '#e6edf3'
C_GRID = '#21262d'
C_BG   = '#161b22'

# ── Plot 1: Training loss ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(C_BG)
ax1.plot(losses, color=C_POS, linewidth=2.5, label='Loss')
ax1.fill_between(range(len(losses)), losses, alpha=0.12, color=C_POS)
ax1.set_title('Training Loss', color=C_TEXT, fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Step', color='#8b949e')
ax1.set_ylabel('Hinge Loss + L2 Reg', color='#8b949e')
ax1.tick_params(colors='#8b949e')
ax1.spines[:].set_color('#30363d')
ax1.grid(True, color=C_GRID, linewidth=0.8)

# ── Plot 2: Accuracy over time ─────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(C_BG)
ax2.plot(accuracies, color='#3fb950', linewidth=2.5, label='Accuracy')
ax2.fill_between(range(len(accuracies)), accuracies, alpha=0.12, color='#3fb950')
ax2.set_ylim(0, 105)
ax2.axhline(100, color='#3fb950', linewidth=0.8, linestyle='--', alpha=0.4)
ax2.set_title('Training Accuracy', color=C_TEXT, fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Step', color='#8b949e')
ax2.set_ylabel('Accuracy (%)', color='#8b949e')
ax2.tick_params(colors='#8b949e')
ax2.spines[:].set_color('#30363d')
ax2.grid(True, color=C_GRID, linewidth=0.8)

# ── Plot 3: Decision boundary ──────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor(C_BG)

# create a fine grid covering the data space
x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 60),
                      np.linspace(y_min, y_max, 60))

# run every grid point through the network
grid_points = np.c_[xx.ravel(), yy.ravel()].tolist()
Z = np.array([n(pt).data for pt in grid_points]).reshape(xx.shape)

# fill the background with the decision regions
ax3.contourf(xx, yy, Z, levels=50,
             cmap=plt.cm.RdYlBu, alpha=0.55)
ax3.contour(xx, yy, Z, levels=[0], colors='white',
            linewidths=2, linestyles='--')  # decision boundary line

# plot the actual data points on top
X_arr = np.array(X)
for label_val, color, marker, name in [
    ( 1.0, C_POS, 'o', 'Class +1'),
    (-1.0, C_NEG, 's', 'Class -1'),
]:
    mask = [yi == label_val for yi in ys]
    ax3.scatter(X_arr[mask, 0], X_arr[mask, 1],
                c=color, s=45, edgecolors='white',
                linewidths=0.6, marker=marker,
                label=name, zorder=5)

ax3.set_title('Decision Boundary', color=C_TEXT, fontsize=13, fontweight='bold', pad=10)
ax3.set_xlabel('x₁', color='#8b949e')
ax3.set_ylabel('x₂', color='#8b949e')
ax3.tick_params(colors='#8b949e')
ax3.spines[:].set_color('#30363d')
ax3.legend(facecolor='#21262d', edgecolor='#30363d',
           labelcolor=C_TEXT, fontsize=9)

plt.suptitle(f'Neural Network from Scratch — Two Moons   |   Final Accuracy: {final_acc*100:.0f}%',
             color=C_TEXT, fontsize=14, fontweight='bold', y=1.02)

plt.savefig('moons_results.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("Plot saved → moons_results.png")
