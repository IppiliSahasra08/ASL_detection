import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
STATS_CSV = "video_stats.csv"
COORDS_CSV = "wrist_coords.csv"

# Load the data
df_stats = pd.read_csv(STATS_CSV)
df_coords = pd.read_csv(COORDS_CSV)

# Set global plotting style for professional reports
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("Generating plots...")

# 1. Temporal Dynamics (Violin Plot of Durations)
plt.figure()
sns.violinplot(data=df_stats, x="class", y="duration_sec", inner="quartile", palette="muted")
plt.title("Distribution of Video Durations per ASL Sign")
plt.xlabel("ASL Sign (Class)")
plt.ylabel("Duration (Seconds)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("1_temporal_durations.png", dpi=300)
plt.close()

# 2. Kinematic Complexity (Average Movement per Sign)
plt.figure()
# Calculate average movement per class and sort
avg_movement = df_stats.groupby("class")["total_movement_pixels"].mean().reset_index()
avg_movement = avg_movement.sort_values("total_movement_pixels", ascending=False)

sns.barplot(data=avg_movement, x="class", y="total_movement_pixels", palette="viridis")
plt.title("Kinematic Complexity: Average Hand Movement per Sign")
plt.xlabel("ASL Sign (Class)")
plt.ylabel("Total Wrist Movement (Pixels)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("2_kinematic_complexity.png", dpi=300)
plt.close()

# 3. Environmental Quality (Brightness Histogram)
plt.figure()
sns.histplot(data=df_stats, x="avg_brightness", bins=30, kde=True, color="skyblue")
plt.title("Dataset Environmental Quality: Average Video Brightness")
plt.xlabel("Average Pixel Intensity (0=Black, 255=White)")
plt.ylabel("Number of Videos")
# Add a vertical line for the mean
plt.axvline(df_stats['avg_brightness'].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean Brightness')
plt.legend()
plt.tight_layout()
plt.savefig("3_environmental_brightness.png", dpi=300)
plt.close()

# 4. Spatial Analysis (2D Density Heatmap of Signing Space)
plt.figure(figsize=(8, 8))
# Note: Inverted Y axis because image coordinates start 0 at the top
sns.kdeplot(data=df_coords, x="x", y="y", cmap="mako", fill=True, thresh=0.05, levels=20)
plt.gca().invert_yaxis() 
plt.title("ASL Spatial Density: Common Signing Space (Wrist Coordinates)")
plt.xlabel("X Coordinate (Pixels)")
plt.ylabel("Y Coordinate (Pixels)")
plt.tight_layout()
plt.savefig("4_spatial_heatmap.png", dpi=300)
plt.close()

print("Done! Look for the 4 PNG files in your folder.")
