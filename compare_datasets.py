import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION & LOAD DATA
# ==========================================
FILE_1 = "/home/psxkf4/IsaacLab/source/collected_data/data_tb-3_ta57_emavel1.0_velstd0.0_broad.csv"
FILE_2 = "/home/psxkf4/IsaacLab/source/collected_data/data_trans_cube.csv"

# Sequence length for time-series extraction
SEQ_LEN = 60

print("Loading datasets...")
df1 = pd.read_csv(FILE_1)
df2 = pd.read_csv(FILE_2)

# Assign labels for comparison
df1['dataset_source'] = 'World frame IsaacLab Dataset'
df2['dataset_source'] = 'Local frame IsaacLab Dataset'

# Combine for joint visualization
df_combined = pd.concat([df1, df2], ignore_index=True)

# ==========================================
# 2. MACRO DATASET INSPECTION (Terminal)
# ==========================================
def print_summary(df, name):
    print(f"\n{'='*50}")
    print(f"DATASET SUMMARY: {name}")
    print(f"{'='*50}")
    print(f"Total Trajectories (Rows): {len(df)}")
    print(f"Total Features (Columns):  {df.shape[1]}")
    
    print("\n[Missing Values]")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("No missing values detected.")
    else:
        print(missing)
        
    print("\n[Ground Truth Statistics]")
    if 'gt_mass' in df.columns and 'gt_mu' in df.columns:
        display_cols = ['gt_mass', 'gt_mu', 'start_t']
        print(df[display_cols].describe().round(4))
    else:
        print("Target columns (gt_mass, gt_mu) not found.")

print_summary(df1, "Dataset 1 (World frame)")
print_summary(df2, "Dataset 2 (Local frame)")

# Compare Columns
cols1 = set(df1.columns)
cols2 = set(df2.columns)
common_cols = cols1.intersection(cols2)
only_in_1 = cols1 - cols2
only_in_2 = cols2 - cols1

print(f"\n{'='*50}")
print("COLUMN COMPARISON")
print(f"{'='*50}")
print(f"Common columns: {len(common_cols)}")
if only_in_1:
    print(f"Columns ONLY in Dataset 1: {only_in_1}")
if only_in_2:
    print(f"Columns ONLY in Dataset 2: {only_in_2}")

# ==========================================
# 3. ENHANCED VELOCITY ANALYSIS
# ==========================================
def analyze_velocity_stats(df, name):
    print(f"\n[Velocity Stats Analysis: {name}]")
    vel_cols = [c for c in df.columns if "input_vel_" in c]
    if vel_cols:
        stats = df[vel_cols].agg(['mean', 'std']).T
        print(stats.head(5).to_string())
        print("...")
        print(stats.tail(5).to_string())
        
        global_mean = df[vel_cols].mean().mean()
        global_std = df[vel_cols].std().mean()
        print(f"\nGlobal Velocity Mean: {global_mean:.4f}")
        print(f"Global Velocity Std:  {global_std:.4f}")
    else:
        print("No columns matching 'input_vel_' found.")

analyze_velocity_stats(df1, "Dataset 1")
analyze_velocity_stats(df2, "Dataset 2")

# ==========================================
# 4. VISUAL INSPECTION (DISTRIBUTIONS)
# ==========================================
sns.set_theme(style="whitegrid")

# Plot 1: Target Properties Distribution (Mass & Friction)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(data=df_combined, x='gt_mass', hue='dataset_source', 
             kde=True, element="step", stat="density", common_norm=False, ax=axes[0])
axes[0].set_title("Distribution of Ground Truth Mass")
axes[0].set_xlabel("Mass (kg)")

sns.histplot(data=df_combined, x='gt_mu', hue='dataset_source', 
             kde=True, element="step", stat="density", common_norm=False, ax=axes[1])
axes[1].set_title("Distribution of Ground Truth Friction (Mu)")
axes[1].set_xlabel("Friction Coefficient")

plt.tight_layout()
plt.show()

# Plot 2: Domain Space 2D Density (Mass vs Mu)
g = sns.JointGrid(data=df_combined, x="gt_mass", y="gt_mu", hue="dataset_source", height=8)
g.plot_joint(sns.scatterplot, s=15, alpha=0.5, edgecolor="none")
g.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)
g.figure.suptitle("Physics Domain Coverage (Mass vs Friction)", y=1.02)
plt.show()

# ==========================================
# 5. VISUAL INSPECTION (TIME SERIES)
# ==========================================
vel_cols = sorted([c for c in common_cols if "input_vel_" in c], 
                  key=lambda x: int(x.split('_')[-1]))
                  
if len(vel_cols) >= SEQ_LEN:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    sample_size = min(50, len(df1), len(df2))
    sample1 = df1.sample(n=sample_size, random_state=42)[vel_cols[:SEQ_LEN]].values
    sample2 = df2.sample(n=sample_size, random_state=42)[vel_cols[:SEQ_LEN]].values
    
    time_steps = np.arange(SEQ_LEN)
    
    for i in range(sample_size):
        axes[0].plot(time_steps, sample1[i], color='blue', alpha=0.1)
        axes[1].plot(time_steps, sample2[i], color='red', alpha=0.1)
        
    axes[0].plot(time_steps, sample1.mean(axis=0), color='black', linewidth=2, label='Mean Velocity')
    axes[1].plot(time_steps, sample2.mean(axis=0), color='black', linewidth=2, label='Mean Velocity')
    
    axes[0].set_title(f"Dataset 1: Kinematic Velocity Profiles ({sample_size} samples)")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Velocity")
    axes[0].legend()
    
    axes[1].set_title(f"Dataset 2: Kinematic Velocity Profiles ({sample_size} samples)")
    axes[1].set_xlabel("Time Step")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
else:
    print("\n[Notice] Velocity columns not found or insufficient sequence length. Skipping time-series plot.")

# ==========================================
# 5.1 ALIGNED GROUND TRUTH VS VELOCITY COMPARISON
# ==========================================
def compare_aligned_trajectories(df1, df2, df_combined, vel_cols, num_samples=5):
    print(f"\n{'='*50}")
    print("ALIGNED TRAJECTORY COMPARISON (Nearest Neighbors)")
    print(f"{'='*50}")

    if len(vel_cols) < SEQ_LEN:
        print("Insufficient velocity columns for time-series extraction.")
        return

    mass_min, mass_max = df_combined['gt_mass'].min(), df_combined['gt_mass'].max()
    mu_min, mu_max = df_combined['gt_mu'].min(), df_combined['gt_mu'].max()

    def get_nearest_neighbor(df, target_mass, target_mu):
        norm_mass_diff = (df['gt_mass'] - target_mass) / (mass_max - mass_min)
        norm_mu_diff = (df['gt_mu'] - target_mu) / (mu_max - mu_min)
        distances = np.sqrt(norm_mass_diff**2 + norm_mu_diff**2)
        
        closest_idx = distances.idxmin()
        actual_mass = df.loc[closest_idx, 'gt_mass']
        actual_mu = df.loc[closest_idx, 'gt_mu']
        vel_seq = np.array(df.loc[closest_idx, vel_cols[:SEQ_LEN]].values, dtype=float)
        
        return actual_mass, actual_mu, vel_seq

    anchor_samples = df_combined.sample(n=num_samples, random_state=101)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples), sharex=True)
    if num_samples == 1:
        axes = [axes]
    time_steps = np.arange(SEQ_LEN)

    for i, (idx, row) in enumerate(anchor_samples.iterrows()):
        target_mass = row['gt_mass']
        target_mu = row['gt_mu']
        
        print(f"\n--- Anchor Target {i+1} | Mass: {target_mass:.4f}, Mu: {target_mu:.4f} ---")

        m1, mu1, seq1 = get_nearest_neighbor(df1, target_mass, target_mu)
        print(f"Dataset 1 Match | Mass: {m1:.4f}, Mu: {mu1:.4f}")

        m2, mu2, seq2 = get_nearest_neighbor(df2, target_mass, target_mu)
        print(f"Dataset 2 Match | Mass: {m2:.4f}, Mu: {mu2:.4f}")

        ax = axes[i]
        ax.plot(time_steps, seq1, label=f"DS1 (Prev): m={m1:.2f}, mu={mu1:.2f}", color='blue', linewidth=2)
        ax.plot(time_steps, seq2, label=f"DS2 (Curr): m={m2:.2f}, mu={mu2:.2f}", color='red', linewidth=2, linestyle='--')
        
        ax.set_title(f"Target Anchor: Mass={target_mass:.2f}kg, Mu={target_mu:.2f}")
        ax.set_ylabel("Input Velocity")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    plt.show()

if 'vel_cols' in locals():
    compare_aligned_trajectories(df1, df2, df_combined, vel_cols, num_samples=20)

print("\nInspection complete.")