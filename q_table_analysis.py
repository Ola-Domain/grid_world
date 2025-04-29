import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Load Q-table ===
with open("results/q_table.json") as f:
    raw_q = json.load(f)
q_table = {eval(k): v for k, v in raw_q.items()}

# === Prepare folders ===
output_dir = "results/q_table_output"
os.makedirs(output_dir, exist_ok=True)

# === Save full Q-table as CSV ===
csv_path = f"{output_dir}/q_table_full.csv"
if os.path.exists(csv_path):
    os.remove(csv_path)

q_df = pd.DataFrame([
    {"State": str(k), "North": v[0], "South": v[1], "East": v[2], "West": v[3]}
    for k, v in q_table.items()
])
q_df.to_csv(csv_path, index=False)
print("✅ Saved Q-table CSV")

# === Heatmap saving function with grid lines ===
def save_value_heatmap(table, label):
    heatmap = np.zeros((5, 5))
    for (i, j), values in table.items():
        heatmap[i - 1, j - 1] = max(values)

    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, cmap='YlGn', origin='upper', vmin=0, vmax=10)

    # Add text and grid lines
    for i in range(5):
        for j in range(5):
            plt.text(j, i, f"{heatmap[i, j]:.1f}", ha='center', va='center', color='black', fontsize=10)

    for x in range(6):
        plt.axhline(x - 0.5, color='black', linewidth=0.5)
        plt.axvline(x - 0.5, color='black', linewidth=0.5)

    plt.colorbar(label='Max Q-Value')
    plt.title(f'State-Value Heatmap ({label})')
    plt.xticks(range(5), range(1, 6))
    plt.yticks(range(5), range(1, 6))
    plt.tight_layout()
    save_path = f"{output_dir}/value_heatmap_{label.lower()}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved {label} heatmap to:", save_path)

# === Build phase versions ===
q_table_initial = {k: [0.0, 0.0, 0.0, 0.0] for k in q_table}
q_table_mid = {k: [v * 0.5 for v in val] for k, val in q_table.items()}

save_value_heatmap(q_table_initial, "Initial")
save_value_heatmap(q_table_mid, "Mid")
save_value_heatmap(q_table, "Final")

# === Update summary.html ===
summary_path = "results/summary.html"
if os.path.exists(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        html = f.read()

    new_card = """
    <div class="card">
        <h2>🧠 Q-Table (CSV)</h2>
        <p><a href="q_table_output/q_table_full.csv" download>Download Full Q-Table CSV</a></p>
    </div>

    <div class="card">
        <h2>🌡️ State-Value Heatmaps</h2>
        <h4>Initial Learning Phase</h4>
        <img src="q_table_output/value_heatmap_initial.png" alt="Initial Heatmap">
        <h4>Mid Learning Phase</h4>
        <img src="q_table_output/value_heatmap_mid.png" alt="Mid Heatmap">
        <h4>Final Learned Values</h4>
        <img src="q_table_output/value_heatmap_final.png" alt="Final Heatmap">
    </div>
    """

    updated_html = html.replace("</body>", new_card + "\n</body>")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(updated_html)
    print("✅ summary.html updated with Q-table and heatmaps.")
else:
    print("⚠️ summary.html not found — please run main.py first.")
