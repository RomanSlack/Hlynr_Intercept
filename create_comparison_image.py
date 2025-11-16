#!/usr/bin/env python3
"""
Create a simple, clean comparison of PPO vs HRL.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure with dark background
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
fig.patch.set_facecolor('#1a1a1a')

# PPO side (left)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_facecolor('#1a1a1a')

# Title
ax1.text(5, 9, 'FLAT PPO', ha='center', va='top',
         fontsize=24, fontweight='bold', color='#ff6b6b')

# Stats
stats_ppo = [
    ('Training:', '10,000,000 steps'),
    ('Time:', 'MONTHS'),
    ('Success:', '70%'),
    ('Miss:', '199m'),
]

y_pos = 7
for label, value in stats_ppo:
    ax1.text(5, y_pos, label, ha='center', va='top',
             fontsize=14, color='#999', fontweight='bold')
    y_pos -= 0.6
    ax1.text(5, y_pos, value, ha='center', va='top',
             fontsize=20, color='#ff6b6b', fontweight='bold')
    y_pos -= 1.8


# HRL side (right)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_facecolor('#1a1a1a')

# Title
ax2.text(5, 9, 'HRL', ha='center', va='top',
         fontsize=24, fontweight='bold', color='#51cf66')

# Stats
stats_hrl = [
    ('Training:', '310,000 steps'),
    ('Time:', '65 MINUTES'),
    ('Success:', '100%'),
    ('Miss:', '0.4m'),
]

y_pos = 7
for label, value in stats_hrl:
    ax2.text(5, y_pos, label, ha='center', va='top',
             fontsize=14, color='#999', fontweight='bold')
    y_pos -= 0.6
    ax2.text(5, y_pos, value, ha='center', va='top',
             fontsize=20, color='#51cf66', fontweight='bold')
    y_pos -= 1.8

# Bottom text
fig.text(0.5, 0.05, '32x FASTER  •  500x MORE PRECISE',
         ha='center', va='center',
         fontsize=18, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/roman/Hlynr_Intercept/HRL_vs_PPO_comparison.jpg',
            format='jpg', dpi=200, bbox_inches='tight',
            facecolor='#1a1a1a', edgecolor='none')

print("Comparison image saved to: /home/roman/Hlynr_Intercept/HRL_vs_PPO_comparison.jpg")
plt.close()
