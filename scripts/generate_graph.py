import matplotlib.pyplot as plt
import numpy as np

# Data
iterations = ['Original', 'Iter 1\n(Aggressive)', 'Iter 2\n(Skip MLP)', 'Iter 3\n(Adaptive)', 'Iter 4\n(K-Proj v1)', 'Iter 5\n(Final v2)']
cosine = [1.0, 0.102, 0.875, 0.939, 0.996, 0.999]
memory = [140, 328, 98, 105, 98, 91] # Estimated values based on reports

x = np.arange(len(iterations))

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Cosine
color = 'tab:blue'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Cosine Similarity', color=color)
ax1.plot(x, cosine, color=color, marker='o', linewidth=2, label='Cosine Similarity')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 1.1)
ax1.grid(True, alpha=0.3)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Memory (MB)', color=color)  # we already handled the x-label with ax1
ax2.plot(x, memory, color=color, marker='s', linestyle='--', linewidth=2, label='Memory (MB)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 400)

# Title and Layout
plt.title('Compression Progress: Quality vs Efficiency', fontsize=14, fontweight='bold')
plt.xticks(x, iterations, rotation=0)
fig.tight_layout()

# Annotate Final Point
ax1.annotate('Final Policy\n(99.9% Quality, -35% Mem)', xy=(5, 0.999), xytext=(3.5, 0.6),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# Save
output_path = 'docs/compression_progress.png'
plt.savefig(output_path, dpi=300)
print(f"Graph saved to {output_path}")
