import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the sweep results
with open('results/sweep_results.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(results)
print("\n✅ Columns in DataFrame:")
print(df.columns)

# Plot 1: Average Reward vs Test Name
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='test_name', y='avg_reward_last30')
plt.title('Average Reward Last 30 Episodes by Test')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/avg_reward_last30_by_test.png')
plt.close()

# Plot 2: Average Steps vs Test Name
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='test_name', y='avg_steps')
plt.title('Average Steps per Episode by Test')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/avg_steps_by_test.png')
plt.close()

# Plot 3: Final Epsilon vs Test Name
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='test_name', y='final_epsilon')
plt.title('Final Epsilon by Test')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/final_epsilon_by_test.png')
plt.close()

# Plot 4: Average Reward vs Episodes Run
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='episodes_run', y='avg_reward_last30', hue='test_name')
plt.title('Average Reward Last 30 Episodes vs Episodes Run')
plt.tight_layout()
plt.savefig('results/avg_reward_vs_episodes_run.png')
plt.close()

# Plot 5: Heatmap of average reward by alpha and gamma (baseline only)
baseline_df = df[df['test_name'] == 'baseline']
baseline_avg = baseline_df.groupby(['alpha', 'gamma'])['avg_reward_last30'].mean().reset_index()
pivot = baseline_avg.pivot(index='alpha', columns='gamma', values='avg_reward_last30')

plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Avg Reward (Baseline) — Alpha vs Gamma')
plt.savefig('results/baseline_alpha_gamma_heatmap.png')
plt.close()

print("\n✅ All plots saved to 'results' folder.")
