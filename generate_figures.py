"""
Run this script to generate all 4 figures for the research paper.
Place this file in: D:\cybersmart_production\federated_learning\
Run: python generate_figures.py
Output: 4 PNG files saved in the same folder
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['font.family'] = 'DejaVu Sans'

# ── DATA FROM SIMULATION ──────────────────────────────────────────────────────
rounds   = list(range(1, 11))
accuracy = [88.10, 94.57, 95.16, 95.74, 96.15, 96.03, 96.73, 96.91, 96.79, 96.67]
f1       = [0.8642, 0.9427, 0.9495, 0.9556, 0.9600, 0.9588, 0.9663, 0.9683, 0.9668, 0.9656]
auc      = [0.9815, 0.9897, 0.9921, 0.9935, 0.9938, 0.9944, 0.9946, 0.9945, 0.9947, 0.9946]
cent_acc = 96.67

# Per-client final round accuracy
clients     = ['Client 0', 'Client 1', 'Client 2', 'Client 3', 'Client 4']
final_accs  = [95.12, 96.14, 96.14, 95.12, None]   # Client 4 not selected round 10
round10_sel = [True, True, True, True, False]

# Confusion matrix
TP, TN, FP, FN = 799, 858, 13, 44

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Convergence Curve
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax2 = ax1.twinx()

color_acc = '#1F3864'
color_f1  = '#2E75B6'
color_auc = '#C00000'

l1, = ax1.plot(rounds, accuracy, 'o-', color=color_acc, linewidth=2.2,
               markersize=6, label='Accuracy (%)', zorder=3)
l2, = ax1.plot(rounds, [f*100 for f in f1], 's--', color=color_f1, linewidth=1.8,
               markersize=5, label='F1 Score ×100', zorder=3)
ax1.axhline(y=cent_acc, color='gray', linestyle=':', linewidth=1.5,
            label=f'Centralised Baseline ({cent_acc}%)', zorder=2)

l3, = ax2.plot(rounds, auc, '^-.', color=color_auc, linewidth=1.8,
               markersize=5, label='AUC-ROC', zorder=3)

ax1.set_xlabel('Communication Round', fontsize=11)
ax1.set_ylabel('Accuracy / F1×100 (%)', fontsize=11, color='black')
ax2.set_ylabel('AUC-ROC', fontsize=11, color=color_auc)
ax2.tick_params(axis='y', labelcolor=color_auc)
ax2.set_ylim(0.97, 1.001)

ax1.set_xlim(0.5, 10.5)
ax1.set_ylim(84, 100)
ax1.set_xticks(rounds)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')

lines = [l1, l2, ax1.get_lines()[2], l3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right', fontsize=9, framealpha=0.9)

ax1.set_title('Figure 1: Global Model Convergence over Communication Rounds',
              fontsize=11, fontweight='bold', pad=12)

plt.tight_layout()
path1 = os.path.join(OUT, 'fig1_convergence.png')
plt.savefig(path1, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {path1}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — System Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

def box(ax, x, y, w, h, text, lines, facecolor, edgecolor, fontsize=8.5):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08", facecolor=facecolor,
        edgecolor=edgecolor, linewidth=1.8, zorder=3)
    ax.add_patch(rect)
    full = text + ('\n' + '\n'.join(lines) if lines else '')
    ax.text(x, y, full, ha='center', va='center',
            fontsize=fontsize, fontweight='bold' if not lines else 'normal',
            color='white' if facecolor in ['#1F3864','#2E75B6','#C00000'] else '#1F3864',
            zorder=4, multialignment='center',
            linespacing=1.4)

def arrow(ax, x1, y1, x2, y2, label='', color='#555555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.6),
        zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.05, my+0.12, label, fontsize=7.5, color=color,
                ha='center', zorder=6, style='italic')

# Mobile clients (left)
client_ys = [4.8, 3.8, 2.8, 1.8, 0.8]
client_labels = ['Client 1\n~1,944 URLs', 'Client 2\n~1,943 URLs',
                 'Client 3\n~1,943 URLs', 'Client 4\n~1,943 URLs',
                 'Client 5\n~1,943 URLs']
for cy, cl in zip(client_ys, client_labels):
    box(ax, 1.3, cy, 1.8, 0.7, cl, [], '#E8F0FB', '#2E75B6', fontsize=8)

# Local training box inside each client indicator
ax.text(1.3, 5.55, 'Mobile Clients (Local Training)', ha='center',
        fontsize=9, fontweight='bold', color='#1F3864')

# Arrows: clients → server
for cy in client_ys:
    arrow(ax, 2.25, cy, 3.5, 2.9, color='#2E75B6')

# FedAvg Server (centre)
box(ax, 5.0, 2.9, 2.4, 1.8,
    'Federated Server\n(FedAvg Aggregation)',
    ['① Select 80% clients',
     '② Broadcast global weights',
     '③ Collect local updates',
     '④ Weighted average'],
    '#1F3864', '#1F3864', fontsize=8)

# Arrow: server → clients (broadcast)
for cy in client_ys:
    arrow(ax, 3.5, 2.9, 2.25, cy, color='#C00000')

ax.text(2.9, 5.55, 'Weight Updates →', ha='center', fontsize=8,
        color='#2E75B6', style='italic')
ax.text(2.9, 5.25, '← Global Model', ha='center', fontsize=8,
        color='#C00000', style='italic')

# Arrow: server → global model
arrow(ax, 6.2, 2.9, 7.2, 2.9, color='#1F3864')

# Global model
box(ax, 8.1, 2.9, 1.6, 1.1,
    'Global\nPhishNet Model',
    ['96.67% Acc', 'AUC 0.9946'],
    '#2E75B6', '#1F3864', fontsize=8)

# Arrow down to inference
arrow(ax, 8.1, 2.35, 8.1, 1.65, color='#1F3864')

# Inference pipeline
box(ax, 8.1, 1.2, 1.6, 0.85,
    'Inference Pipeline',
    ['Whitelist → Neural', 'Threshold = 0.55'],
    '#F2F2F2', '#2E75B6', fontsize=7.5)

# Privacy note
ax.text(5.0, 0.25,
        'Privacy Guarantee: Raw URL data never leaves the device — only model weight updates are transmitted.',
        ha='center', fontsize=8, color='#555555', style='italic')

ax.set_title('Figure 2: Federated Learning System Architecture',
             fontsize=11, fontweight='bold', pad=10)
plt.tight_layout()
path2 = os.path.join(OUT, 'fig2_architecture.png')
plt.savefig(path2, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {path2}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Confusion Matrix
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 4.5))

cm = np.array([[TN, FP], [FN, TP]])
labels_cm = np.array([
    [f'TN\n{TN}\n(Legitimate → Safe)', f'FP\n{FP}\n(Legitimate → Phishing)'],
    [f'FN\n{FN}\n(Phishing → Safe)',   f'TP\n{TP}\n(Phishing → Phishing)'],
])

colors = np.array([
    ['#D5E8D4', '#FFE6CC'],
    ['#FFE6CC', '#DAE8FC'],
])

for i in range(2):
    for j in range(2):
        rect = mpatches.FancyBboxPatch((j*3 + 0.1, (1-i)*2.2 + 0.1), 2.7, 2.0,
            boxstyle="round,pad=0.05", facecolor=colors[i][j],
            edgecolor='#555555', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(j*3 + 1.45, (1-i)*2.2 + 1.1, labels_cm[i][j],
                ha='center', va='center', fontsize=10.5,
                fontweight='bold', multialignment='center',
                color='#1F3864')

ax.set_xlim(0, 6.2)
ax.set_ylim(0, 4.5)
ax.axis('off')

ax.text(1.5, 4.3, 'Predicted: Legitimate', ha='center', fontsize=10, fontweight='bold', color='#1F3864')
ax.text(4.5, 4.3, 'Predicted: Phishing',   ha='center', fontsize=10, fontweight='bold', color='#1F3864')
ax.text(-0.7, 3.2, 'Actual:\nLegitimate', ha='center', fontsize=9.5, fontweight='bold', color='#1F3864', rotation=0, multialignment='center')
ax.text(-0.7, 1.1, 'Actual:\nPhishing',   ha='center', fontsize=9.5, fontweight='bold', color='#1F3864', rotation=0, multialignment='center')

total = TP + TN + FP + FN
ax.text(3.1, -0.15,
        f'Accuracy: {(TP+TN)/total*100:.2f}%   Precision: {TP/(TP+FP)*100:.2f}%   '
        f'Recall: {TP/(TP+FN)*100:.2f}%   F1: {2*TP/(2*TP+FP+FN):.4f}',
        ha='center', fontsize=8.5, color='#444444', style='italic')

ax.set_title('Figure 3: Confusion Matrix — Final Federated Global Model\n(Test Set: 1,714 samples)',
             fontsize=10.5, fontweight='bold', pad=6)

fig.patch.set_facecolor('white')
plt.tight_layout()
path3 = os.path.join(OUT, 'fig3_confusion_matrix.png')
plt.savefig(path3, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {path3}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Per-Client Local Accuracy Across Rounds
# ═══════════════════════════════════════════════════════════════════════════════
# Raw per-client accuracy from simulation output (None = not selected that round)
client_data = {
    'Client 0': [0.8509, 0.9280, 0.9280, 0.9460, 0.9383, 0.9512, 0.9434, 0.9512, 0.9486, 0.9512],
    'Client 1': [0.8972, 0.9512, 0.9614, None,   0.9666, 0.9563, None,   0.9589, 0.9640, 0.9614],
    'Client 2': [0.8843, 0.9383, 0.9486, 0.9486, 0.9589, None,   0.9614, 0.9666, 0.9589, 0.9614],
    'Client 3': [None,   None,   None,   0.9589, 0.9563, 0.9563, 0.9614, 0.9589, 0.9589, 0.9512],
    'Client 4': [0.8869, 0.9306, 0.9460, 0.9434, None,   0.9537, 0.9537, None,   None,   None  ],
}

colors_c = ['#1F3864', '#2E75B6', '#C00000', '#70AD47', '#ED7D31']
markers  = ['o', 's', '^', 'D', 'v']

fig, ax = plt.subplots(figsize=(8, 4.5))

for (name, vals), col, mk in zip(client_data.items(), colors_c, markers):
    xs = [r for r, v in zip(rounds, vals) if v is not None]
    ys = [v*100 for v in vals if v is not None]
    ax.plot(xs, ys, marker=mk, color=col, linewidth=1.8,
            markersize=6, label=name, zorder=3, linestyle='--', alpha=0.85)

ax.axhline(y=96.67, color='black', linestyle='-', linewidth=1.8,
           label='Global Model (96.67%)', zorder=4)
ax.set_xlabel('Communication Round', fontsize=11)
ax.set_ylabel('Local Validation Accuracy (%)', fontsize=11)
ax.set_xlim(0.5, 10.5)
ax.set_ylim(83, 100)
ax.set_xticks(rounds)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')
ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
ax.set_title('Figure 4: Per-Client Local Validation Accuracy Across Communication Rounds\n'
             '(Dashed = client selected that round; gaps = client not selected)',
             fontsize=10.5, fontweight='bold', pad=10)

plt.tight_layout()
path4 = os.path.join(OUT, 'fig4_client_accuracy.png')
plt.savefig(path4, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {path4}")

print("\n✅ All 4 figures generated successfully!")
print("Files saved:")
for p in [path1, path2, path3, path4]:
    print(f"  {p}")
print("\nSend these 4 PNG files and I will embed them into the final paper.")
