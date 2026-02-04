#!/usr/bin/env python3
"""Generate README visualization from garbling game results."""

import matplotlib.pyplot as plt
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

# Find most recent run
results_dir = Path("run_results")
run_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
if not run_dirs:
    print("No runs found")
    exit(1)

latest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
print(f"Visualizing: {latest_run.name}")

# Load results
with open(latest_run / "results.json") as f:
    data = json.load(f)

with open(latest_run / "metadata.json") as f:
    metadata = json.load(f)

# Extract results structure
results = data.get("summary", {})
history = data.get("history", [])
num_rounds = results.get("total_rounds", len(history))

# Calculate cumulative totals from history
sender_totals = []
receiver_totals = []
cumulative_sender = 0
cumulative_receiver = 0
for round_data in history:
    cumulative_sender += round_data.get("sender_payoff", 0)
    cumulative_receiver += round_data.get("receiver_payoff", 0)
    sender_totals.append(cumulative_sender)
    receiver_totals.append(cumulative_receiver)

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Garbling Economics Game Results\n{metadata.get("name", "unnamed")} - {num_rounds} rounds',
             fontsize=16, fontweight='bold', y=0.98)

# 1. Total Values Over Time
rounds = list(range(1, num_rounds + 1))

ax1.plot(rounds, sender_totals, 'o-', color='#667eea', linewidth=2.5,
         markersize=6, label='Sender', alpha=0.8)
ax1.plot(rounds, receiver_totals, 's-', color='#4facfe', linewidth=2.5,
         markersize=6, label='Receiver', alpha=0.8)
ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
ax1.set_ylabel('Total Value', fontsize=12, fontweight='bold')
ax1.set_title('Cumulative Values Over Time', fontsize=14, fontweight='bold', pad=10)
ax1.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax1.grid(alpha=0.3)

# 2. Communication Decisions
buy_count = sum(1 for round_data in history if round_data.get("action") == "BUY")
no_buy_count = len(history) - buy_count
buy_rate = (buy_count / len(history)) * 100 if history else 0

colors = ['#10b981', '#ef4444']
explode = (0.05, 0)
wedges, texts, autotexts = ax2.pie(
    [buy_count, no_buy_count],
    labels=['Buy', 'No Buy'],
    autopct='%1.1f%%',
    colors=colors,
    explode=explode,
    shadow=True,
    startangle=90,
    textprops={'fontsize': 12, 'fontweight': 'bold'}
)
ax2.set_title(f'Receiver Decisions\n({buy_count} buy, {no_buy_count} no buy)',
              fontsize=14, fontweight='bold', pad=10)

# Make percentage text white for visibility
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 3. Informativeness Distribution
informativeness_scores = [round_data.get("garbling_info", 0) for round_data in history]
avg_info = results.get("avg_informativeness", sum(informativeness_scores) / len(informativeness_scores) if informativeness_scores else 0)
ax3.hist(informativeness_scores, bins=20, color='#f093fb', alpha=0.8,
         edgecolor='black', linewidth=1.2)
ax3.axvline(avg_info, color='red', linestyle='--',
            linewidth=2.5, label=f'Avg: {avg_info:.3f}')
ax3.set_xlabel('Informativeness Score', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Message Informativeness Distribution', fontsize=14, fontweight='bold', pad=10)
ax3.legend(fontsize=11, framealpha=0.9)
ax3.grid(axis='y', alpha=0.3)

# 4. Game Summary
ax4.axis('off')

summary_text = f"""
GAME SUMMARY

Rounds: {num_rounds}

Sender Total: {results.get('sender_total', 0):.2f}

Receiver Total: {results.get('receiver_total', 0):.2f}

Buy Rate: {buy_rate:.1f}%

Avg Informativeness: {avg_info:.3f}

Correlation (Info vs Buy):
{results.get('informativeness_buy_correlation', 'N/A')}
"""

ax4.text(0.5, 0.5, summary_text,
         transform=ax4.transAxes,
         fontsize=13,
         verticalalignment='center',
         horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8, pad=1),
         family='monospace',
         fontweight='bold')

plt.tight_layout()

# Save as JPG
output_path = Path("images/simulation_results.jpg")
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, format='jpg', dpi=300, bbox_inches='tight')
print(f"Saved visualization to {output_path}")

plt.close()
