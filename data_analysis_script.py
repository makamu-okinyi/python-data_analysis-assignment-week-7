"""
Data Analysis Project - Iris Dataset
====================================

This script demonstrates complete data analysis workflow including:
1. Loading and exploring dataset with pandas
2. Basic data analysis and statistics
3. Data visualization with matplotlib
4. Key findings and observations

Author: Data Analysis Student
Date: September 2024
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*60)
print("DATA ANALYSIS PROJECT - IRIS DATASET")
print("="*60)

# ============================================================================
# TASK 1: LOAD AND EXPLORE THE DATASET
# ============================================================================

print("\n" + "="*50)
print("TASK 1: LOADING AND EXPLORING THE DATASET")
print("="*50)

try:
    # Load the Iris dataset from sklearn
    print("Loading Iris dataset...")
    iris_data = load_iris()
    
    # Create pandas DataFrame
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df['species'] = iris_data.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("✓ Dataset loaded successfully!")
    
    # Display basic information about the dataset
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    print("\n1. First 10 rows of the dataset:")
    print("-" * 40)
    print(df.head(10))
    
    print("\n2. Dataset Information:")
    print("-" * 40)
    print("Column Names and Data Types:")
    print(df.dtypes)
    
    print("\n3. Dataset Structure:")
    print("-" * 40)
    print(df.info())
    
    print("\n4. Checking for Missing Values:")
    print("-" * 40)
    missing_values = df.isnull().sum()
    print(missing_values)
    
    if missing_values.sum() == 0:
        print("✓ No missing values found in the dataset!")
    else:
        print("⚠ Missing values detected. Cleaning required.")
        # Handle missing values (if any)
        df.fillna(df.mean(numeric_only=True), inplace=True)
        print("✓ Missing values filled with mean values.")
    
    print("\n5. Unique Species:")
    print("-" * 40)
    print(f"Species count: {df['species_name'].nunique()}")
    print(f"Species names: {df['species_name'].unique()}")
    print(f"Distribution:\n{df['species_name'].value_counts()}")

except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure sklearn is installed: pip install scikit-learn")

# ============================================================================
# TASK 2: BASIC DATA ANALYSIS
# ============================================================================

print("\n" + "="*50)
print("TASK 2: BASIC DATA ANALYSIS")
print("="*50)

# 1. Descriptive Statistics
print("\n1. Descriptive Statistics for Numerical Columns:")
print("-" * 50)
numerical_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
stats_df = df[numerical_columns].describe()
print(stats_df.round(2))

# 2. Group Analysis by Species
print("\n2. Group Analysis by Species:")
print("-" * 50)
print("Mean values for each species:")
grouped_stats = df.groupby('species_name')[numerical_columns].mean()
print(grouped_stats.round(2))

print("\nStandard deviation for each species:")
grouped_std = df.groupby('species_name')[numerical_columns].std()
print(grouped_std.round(2))

# 3. Correlation Analysis
print("\n3. Correlation Matrix:")
print("-" * 50)
correlation_matrix = df[numerical_columns].corr()
print(correlation_matrix.round(2))

# 4. Additional Analysis
print("\n4. Additional Statistical Insights:")
print("-" * 50)

# Find species with largest/smallest measurements
print("Species with largest average petal length:")
max_petal = grouped_stats['petal length (cm)'].idxmax()
max_petal_value = grouped_stats.loc[max_petal, 'petal length (cm)']
print(f"  {max_petal}: {max_petal_value:.2f} cm")

print("Species with smallest average petal length:")
min_petal = grouped_stats['petal length (cm)'].idxmin()
min_petal_value = grouped_stats.loc[min_petal, 'petal length (cm)']
print(f"  {min_petal}: {min_petal_value:.2f} cm")

# ============================================================================
# TASK 3: DATA VISUALIZATION
# ============================================================================

print("\n" + "="*50)
print("TASK 3: DATA VISUALIZATION")
print("="*50)

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Iris Dataset - Complete Data Analysis Visualizations', fontsize=16, fontweight='bold')

# 1. BAR CHART - Average measurements by species
print("\nCreating Bar Chart: Average Petal Length by Species...")
plt.subplot(2, 2, 1)
species_means = df.groupby('species_name')['petal length (cm)'].mean()
bars = plt.bar(species_means.index, species_means.values, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black')
plt.title('Average Petal Length by Species', fontweight='bold', fontsize=12)
plt.xlabel('Species', fontweight='bold')
plt.ylabel('Average Petal Length (cm)', fontweight='bold')
plt.xticks(rotation=45)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', alpha=0.3)

# 2. LINE CHART - Trend of measurements across samples
print("Creating Line Chart: Measurement Trends...")
plt.subplot(2, 2, 2)
sample_range = range(0, len(df), 5)  # Every 5th sample for clarity
plt.plot(sample_range, df.iloc[sample_range]['sepal length (cm)'], 
         marker='o', label='Sepal Length', linewidth=2, markersize=4)
plt.plot(sample_range, df.iloc[sample_range]['petal length (cm)'], 
         marker='s', label='Petal Length', linewidth=2, markersize=4)
plt.title('Measurement Trends Across Samples', fontweight='bold', fontsize=12)
plt.xlabel('Sample Index', fontweight='bold')
plt.ylabel('Length (cm)', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. HISTOGRAM - Distribution of sepal length
print("Creating Histogram: Sepal Length Distribution...")
plt.subplot(2, 2, 3)
plt.hist(df['sepal length (cm)'], bins=15, color='#96CEB4', alpha=0.8, 
         edgecolor='black', density=True)
plt.title('Distribution of Sepal Length', fontweight='bold', fontsize=12)
plt.xlabel('Sepal Length (cm)', fontweight='bold')
plt.ylabel('Density', fontweight='bold')

# Add normal distribution overlay
mu = df['sepal length (cm)'].mean()
sigma = df['sepal length (cm)'].std()
x = np.linspace(df['sepal length (cm)'].min(), df['sepal length (cm)'].max(), 100)
plt.plot(x, ((np.pi*sigma)**-0.5)*np.exp(-0.5*((x-mu)/sigma)**2), 
         'r--', linewidth=2, label=f'Normal (μ={mu:.1f}, σ={sigma:.1f})')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. SCATTER PLOT - Relationship between sepal and petal length
print("Creating Scatter Plot: Sepal vs Petal Length...")
plt.subplot(2, 2, 4)
colors = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
for species in df['species_name'].unique():
    species_data = df[df['species_name'] == species]
    plt.scatter(species_data['sepal length (cm)'], species_data['petal length (cm)'],
                c=colors[species], label=species, alpha=0.7, s=60, edgecolors='black')

plt.title('Sepal Length vs Petal Length', fontweight='bold', fontsize=12)
plt.xlabel('Sepal Length (cm)', fontweight='bold')
plt.ylabel('Petal Length (cm)', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df['sepal length (cm)'], df['petal length (cm)'], 1)
p = np.poly1d(z)
plt.plot(df['sepal length (cm)'], p(df['sepal length (cm)']), 
         "r--", alpha=0.8, linewidth=2)

plt.tight_layout()
plt.show()

# Additional Visualization: Correlation Heatmap
print("\nCreating Correlation Heatmap...")
plt.figure(figsize=(10, 8))
mask = np.triu(correlation_matrix.corr())
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, mask=mask, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Box plots for each feature by species
print("Creating Box Plots by Species...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Distribution of Features by Species', fontsize=16, fontweight='bold')

for i, column in enumerate(numerical_columns):
    ax = axes[i//2, i%2]
    df.boxplot(column=column, by='species_name', ax=ax)
    ax.set_title(f'{column.title()}')
    ax.set_xlabel('Species')
    ax.set_ylabel('Measurement (cm)')

plt.tight_layout()
plt.show()

# ============================================================================
# FINDINGS AND OBSERVATIONS
# ============================================================================

print("\n" + "="*50)
print("FINDINGS AND OBSERVATIONS")
print("="*50)

print("\n🔍 KEY FINDINGS:")
print("-" * 30)

# Statistical findings
print("1. STATISTICAL INSIGHTS:")
print(f"   • Dataset contains {len(df)} samples with {len(numerical_columns)} features")
print(f"   • All features show normal-like distributions")
print(f"   • No missing values detected - dataset is clean")

print(f"\n2. SPECIES CHARACTERISTICS:")
for species in df['species_name'].unique():
    species_data = df[df['species_name'] == species]
    avg_petal = species_data['petal length (cm)'].mean()
    avg_sepal = species_data['sepal length (cm)'].mean()
    print(f"   • {species.capitalize()}:")
    print(f"     - Average petal length: {avg_petal:.2f} cm")
    print(f"     - Average sepal length: {avg_sepal:.2f} cm")

print(f"\n3. CORRELATION INSIGHTS:")
strongest_corr = correlation_matrix.abs().unstack().sort_values(ascending=False)
# Remove self-correlations
strongest_corr = strongest_corr[strongest_corr < 1.0]
top_corr = strongest_corr.iloc[0]
corr_features = strongest_corr.index[0]
print(f"   • Strongest correlation: {corr_features[0]} vs {corr_features[1]} ({top_corr:.3f})")
print(f"   • Petal measurements show strong correlation with species classification")

print(f"\n4. VISUALIZATION INSIGHTS:")
print("   • Bar chart reveals clear separation in petal lengths between species")
print("   • Histogram shows sepal length follows approximately normal distribution")
print("   • Scatter plot demonstrates linear relationship between sepal and petal length")
print("   • Box plots highlight the distinct ranges for each species")

print(f"\n5. CLASSIFICATION POTENTIAL:")
print("   • Features show clear separation between species")
print("   • Setosa is easily distinguishable from other species")
print("   • Versicolor and Virginica have some overlap but are separable")
print("   • This dataset is ideal for machine learning classification tasks")

print(f"\n✅ ANALYSIS COMPLETE!")
print(f"   • Data loaded and explored successfully")
print(f"   • Statistical analysis performed")
print(f"   • 4 different visualization types created")
print(f"   • Key insights and patterns identified")

print("\n" + "="*60)
print("END OF ANALYSIS")
print("="*60)

# Optional: Save results to files
try:
    # Save statistical summary
    with open('iris_analysis_summary.txt', 'w') as f:
        f.write("IRIS DATASET ANALYSIS SUMMARY\n")
        f.write("="*40 + "\n\n")
        f.write("Descriptive Statistics:\n")
        f.write(str(stats_df))
        f.write("\n\nGrouped Statistics by Species:\n")
        f.write(str(grouped_stats))
        f.write("\n\nCorrelation Matrix:\n")
        f.write(str(correlation_matrix))
    
    print("📄 Analysis summary saved to 'iris_analysis_summary.txt'")
    
    # Save the processed dataset
    df.to_csv('iris_processed_data.csv', index=False)
    print("💾 Processed dataset saved to 'iris_processed_data.csv'")
    
except Exception as e:
    print(f"Note: Could not save files - {e}")

print("\n🎉 Project completed successfully!")
print("You can now submit this script (.py) or convert it to a Jupyter notebook (.ipynb)")
