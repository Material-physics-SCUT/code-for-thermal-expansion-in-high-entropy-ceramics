import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read Excel file
file_name = "high_entropy_features.xlsx"  # File name
data = pd.read_excel(file_name)

# Extract features and target values
feature_columns = data.columns[1:-1]  # Second column to second last column are features
target_column = data.columns[-1]  # Last column is the target value

# Extract features and target values
features = data[feature_columns]
target = data[target_column]

# Calculate correlation matrix between features
correlation_matrix_features = features.corr()

# Calculate correlation between features and target
correlation_with_target = features.corrwith(target)

# ------------ Heatmap Section ------------ #
# Plot heatmap of feature-to-feature correlations (without annotations)
plt.figure(figsize=(12, 10))  # Adjust heatmap size
sns.heatmap(
    correlation_matrix_features,
    cmap="vlag",  # Updated color style, replaced with softer `vlag` color
    cbar=True,
    square=True
)
plt.title("Feature-to-Feature Correlation Heatmap (No Annotations)", fontsize=18)  # Increased title font size
plt.xticks(rotation=45, fontsize=18)  # Increased x-axis label font size
plt.yticks(rotation=0, fontsize=18)  # Increased y-axis label font size
plt.tight_layout()
plt.savefig("feature_to_feature_correlation_heatmap_no_annotations.png", dpi=300)  # Save image
plt.show()

# ------------ Bar Chart Section ------------ #
# Plot bar chart of feature-to-target correlations
correlation_with_target_sorted = correlation_with_target.sort_values(key=abs, ascending=False)  # Sort by absolute value
plt.figure(figsize=(14, 8))  # Adjust bar chart size
sns.barplot(
    x=correlation_with_target_sorted.index,
    y=correlation_with_target_sorted.values,
    palette="crest"  # Updated color style, replaced with more aesthetic `crest` palette
)
plt.title("Feature-to-Target Correlation", fontsize=18)  # Increased title font size
plt.ylabel("Correlation", fontsize=18)  # Increased y-axis label font size
plt.xlabel("Features", fontsize=18)  # Increased x-axis label font size
plt.xticks(rotation=45, fontsize=14)  # Increased x-axis tick font size
plt.yticks(fontsize=14)  # Increased y-axis tick font size
plt.tight_layout()
plt.savefig("feature_to_target_correlation_barplot.png", dpi=300)  # Save image
plt.show()

# ------------ Output Strong Correlation Feature Combinations ------------ #
# Find strongly correlated feature pairs (absolute value greater than threshold)
threshold = 0.7  # Set correlation threshold
strong_feature_pairs = []
for i in range(len(correlation_matrix_features.columns)):
    for j in range(i + 1, len(correlation_matrix_features.columns)):  # Avoid duplicate comparisons
        feature1 = correlation_matrix_features.columns[i]
        feature2 = correlation_matrix_features.columns[j]
        correlation = correlation_matrix_features.iloc[i, j]
        if abs(correlation) > threshold:
            strong_feature_pairs.append((feature1, feature2, correlation))

# Output results to file
output_file = "output_results.txt"
with open(output_file, "w") as file:
    file.write(f"Feature pairs with correlation absolute value > {threshold}:\n")
    for pair in strong_feature_pairs:
        file.write(f"{pair[0]} and {pair[1]} correlation: {pair[2]:.2f}\n")
    
    file.write("\nTop 8 features with strongest correlation to target:\n")
    top_8_features = correlation_with_target.abs().sort_values(ascending=False).head(18)  # Sort by absolute value and take top 8
    for feature, corr in top_8_features.items():
        file.write(f"{feature} correlation: {corr:.2f}\n")

# Print some information to console
print(f"Feature pairs with correlation absolute value > {threshold} have been saved to {output_file}!")
print(f"Top 8 features with strongest correlation to target have also been saved to {output_file}!")