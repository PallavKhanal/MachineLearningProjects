"""
train.py
--------
Trains a feedforward neural network (built from scratch) to solve XOR.

XOR is the classic proof that neural networks can learn non-linear patterns
that no linear model can — making it the perfect minimal demo.

Truth table:
    [0, 0] → -1  (False XOR False = False)
    [0, 1] →  1  (False XOR True  = True)
    [1, 0] →  1  (True  XOR False = True)
    [1, 1] → -1  (True  XOR True  = False)

Run:
    python train.py
"""

import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fnn import MLP

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)

# ── Dataset: XOR ──────────────────────────────────────────────────────────────
xs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
ys = [-1.0, 1.0, 1.0, -1.0]   # using ±1 (tanh output range) instead of 0/1

# ── Model ─────────────────────────────────────────────────────────────────────
# 2 inputs → hidden layer (4) → hidden layer (4) → 1 output
model = MLP(nin=2, nouts=[4, 4, 1])
print(f"Model: {model}")
print(f"Total trainable parameters: {model.num_params}\n")

# ── Hyperparameters ───────────────────────────────────────────────────────────
LEARNING_RATE = 0.1
STEPS         = 100

# ── Training loop ─────────────────────────────────────────────────────────────
losses = []

for step in range(STEPS):
    # 1. Forward pass — run every input through the network
    ypred = [model(x) for x in xs]

    # 2. Loss — Mean Squared Error: average of (prediction - ground_truth)^2
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)) * (1 / len(ys))

    # 3. Zero gradients — must reset before every backward pass
    for p in model.parameters():
        p.grad = 0.0

    # 4. Backward pass — compute d(loss)/d(weight) for every weight
    loss.backward()

    # 5. Gradient descent — nudge every weight in the direction that reduces loss
    for p in model.parameters():
        p.data -= LEARNING_RATE * p.grad

    losses.append(loss.data)

    if step % 10 == 0 or step == STEPS - 1:
        preds = [round(p.data, 3) for p in ypred]
        print(f"step {step:3d} | loss = {loss.data:.6f} | preds = {preds}")

# ── Final predictions ─────────────────────────────────────────────────────────
print("\n── Final predictions ──────────────────────────────────────")
print(f"{'Input':<12} {'Target':>8} {'Predicted':>12} {'Correct?':>10}")
print("─" * 46)
correct = 0
for x, y in zip(xs, ys):
    pred = model(x).data
    predicted_label = 1.0 if pred > 0 else -1.0
    ok = "✓" if predicted_label == y else "✗"
    if predicted_label == y:
        correct += 1
    print(f"{str(x):<12}  {y:>6.1f}   {pred:>10.4f}  {ok:>8}")
print(f"\nAccuracy: {correct}/{len(xs)} = {correct/len(xs)*100:.0f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5), facecolor='#0d1117')
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# --- Left: Loss curve ---
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#161b22')
ax1.plot(losses, color='#58a6ff', linewidth=2.5, label='MSE Loss')
ax1.fill_between(range(len(losses)), losses, alpha=0.15, color='#58a6ff')
ax1.set_title('Training Loss', color='#e6edf3', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('Training Step', color='#8b949e', fontsize=11)
ax1.set_ylabel('MSE Loss', color='#8b949e', fontsize=11)
ax1.tick_params(colors='#8b949e')
ax1.spines[:].set_color('#30363d')
ax1.grid(True, color='#21262d', linewidth=0.8)
ax1.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')

# --- Right: Predictions vs targets ---
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor('#161b22')

labels     = ['[0,0]', '[0,1]', '[1,0]', '[1,1]']
targets    = ys
predicted  = [model(x).data for x in xs]
x_pos      = range(len(labels))

bars_t = ax2.bar([p - 0.2 for p in x_pos], targets,   width=0.35,
                  color='#3fb950', alpha=0.85, label='Target')
bars_p = ax2.bar([p + 0.2 for p in x_pos], predicted, width=0.35,
                  color='#58a6ff', alpha=0.85, label='Predicted')

ax2.axhline(0, color='#30363d', linewidth=1)
ax2.set_title('Predictions vs Targets (XOR)', color='#e6edf3',
              fontsize=14, fontweight='bold', pad=12)
ax2.set_xticks(list(x_pos))
ax2.set_xticklabels(labels, color='#8b949e')
ax2.set_ylabel('Output value', color='#8b949e', fontsize=11)
ax2.set_ylim(-1.4, 1.4)
ax2.tick_params(colors='#8b949e')
ax2.spines[:].set_color('#30363d')
ax2.grid(True, axis='y', color='#21262d', linewidth=0.8)
ax2.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')

plt.suptitle('Neural Network from Scratch — XOR Problem',
             color='#e6edf3', fontsize=15, fontweight='bold', y=1.02)

plt.savefig('training_results.png',
            dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("\nPlot saved → training_results.png")
