# Standard library imports
import io
import os
import warnings
import traceback
import sys
import pickle


# Data science & ML libraries
import bentoml
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from pycaret.regression import *
from scipy import stats
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

# scikit-learn imports
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, PowerTransformer


GlobalHydra.instance().clear()

initialize(config_path="../configs", job_name="property_prices")

sns.set_palette("husl")
pd.set_option("display.max_columns", None)

warnings.filterwarnings("ignore")

# Read the dataset
df = pd.read_csv(cfg.data.dataset_path)

# Set up MLFlow experiment
mlflow.set_experiment(cfg.experiment.name)
mlflow.start_run(run_name=cfg.experiment.run_name)

# Display basic information about the dataset
# Log dataset shape as parameters
mlflow.log_param("python_version", os.popen('python --version').read().strip())

mlflow.log_param("initial_rows", df.shape[0])
mlflow.log_param("initial_columns", df.shape[1])

mlflow.log_param("columns", list(df.columns))
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
df.info()



df.head().to_csv("sample_data.csv", index=False)
mlflow.log_artifact("sample_data.csv", "data_samples")
df.head()


# Calculate and log missing values information
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_summary = pd.DataFrame(
    {"Missing Values": missing_values, "Percentage": missing_percentage}
).sort_values("Percentage", ascending=False)

print("\nMissing Values Summary:")
print(missing_summary)

# Log missing values statistics
for column in df.columns:
    mlflow.log_metric(f"missing_pct_{column}", missing_percentage[column])

# Log overall missing data metrics
mlflow.log_metric("total_missing_values", missing_values.sum())
mlflow.log_metric("avg_missing_percentage", missing_percentage.mean())

# Save and log the missing values summary
missing_summary.to_csv("missing_values_summary.csv")
mlflow.log_artifact("missing_values_summary.csv", "data_quality")


# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")


# Display basic statistics
print("\nBasic Statistics:")
display(df.describe())


# Define helper function for visualization
def plot_distribution_and_boxplot(df, column):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Distribution plot
    sns.histplot(data=df, x=column, kde=True, ax=ax1)
    ax1.set_title(f'Distribution of {column}')


    # Box plot
    sns.boxplot(data=df, y=column, ax=ax2)
    ax2.set_title(f'Box Plot of {column}')

    plt.tight_layout()
    plt.show()


# Analyze price distribution
plot_distribution_and_boxplot(df, 'Price')

# Display basic statistics for price
price_stats = df['Price'].describe()
print("\nPrice Statistics:")
display(price_stats)

# Log price statistics
for stat_name, stat_value in price_stats.items():
    mlflow.log_metric(f"price_{stat_name.lower().replace('%', '')}", stat_value)

# Log skewness of price
mlflow.log_metric("price_skewness", df["Price"].skew())



# Analyze property size features
size_features = ['BuildingArea', 'Landsize']
for feature in size_features:
    plot_distribution_and_boxplot(df, feature)
    feature_stats = df[feature].describe()
    for stat_name, stat_value in feature_stats.items():
        mlflow.log_metric(f"{feature.lower()}_{stat_name.lower().replace('%', '')}", stat_value)
    # Log skewness
    mlflow.log_metric(f"{feature.lower()}_skewness", df[feature].skew())


# Analyze amenity features
amenity_features = ['Rooms', 'Bedroom2', 'Bathroom', 'Car']
for feature in amenity_features:
    plot_distribution_and_boxplot(df, feature)
    feature_stats = df[feature].describe()
    for stat_name, stat_value in feature_stats.items():
        mlflow.log_metric(f"{feature.lower()}_{stat_name.lower().replace('%', '')}", stat_value)
    # Log skewness
    mlflow.log_metric(f"{feature.lower()}_skewness", df[feature].skew())


# Analyze location features
plot_distribution_and_boxplot(df, 'Distance')
distance_stats = df["Distance"].describe()
for stat_name, stat_value in distance_stats.items():
    mlflow.log_metric(f"distance_{stat_name.lower().replace('%', '')}", stat_value)
mlflow.log_metric("distance_skewness", df["Distance"].skew())


plot_distribution_and_boxplot(df, 'YearBuilt')
yearbuilt_stats = df["YearBuilt"].describe()
for stat_name, stat_value in yearbuilt_stats.items():
    mlflow.log_metric(f"yearbuilt_{stat_name.lower().replace('%', '')}", stat_value)
mlflow.log_metric("yearbuilt_skewness", df["YearBuilt"].skew())



def calculate_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]

    return {
        "Column": column,
        "Outliers Count": len(outliers),
        "Outliers Percentage": (len(outliers) / len(df)) * 100,
        "Min": outliers.min() if len(outliers) > 0 else None,
        "Max": outliers.max() if len(outliers) > 0 else None,
    }

numerical_columns = [
    "Price", "Rooms", "Distance", "Bedroom2", "Bathroom", 
    "Car", "Landsize", "BuildingArea", "YearBuilt",
]

outliers_summary = pd.DataFrame(
    [calculate_outliers(df, col) for col in numerical_columns]
)

display(outliers_summary)

# Log outliers information
for _, row in outliers_summary.iterrows():
    column = row['Column']
    mlflow.log_metric(f"{column.lower()}_outliers_count", row['Outliers Count'])
    mlflow.log_metric(f"{column.lower()}_outliers_percentage", row['Outliers Percentage'])

# Save and log the outliers summary
outliers_summary.to_csv("outliers_summary.csv", index=False)
mlflow.log_artifact("outliers_summary.csv")


# Calculate and display skewness
skewness = df[numerical_columns].skew()
print("\nSkewness Analysis:")
display(pd.DataFrame({'Skewness': skewness}))
# Log skewness values
for col, skew_value in skewness.items():
    mlflow.log_metric(f"{col.lower()}_skewness", skew_value)


def plot_categorical_distributions(df, categorical_cols, figsize=(18, 15)):
    """
    Create a compact visualization of categorical variables showing:
    1. Distribution of categories (count)
    2. Average price by category
    3. Box plot of price distribution by category (for variables with few categories)
    """
    # Calculate number of rows needed (2 plots per categorical variable)
    n_cols = 2
    n_rows = len(categorical_cols)

    fig = plt.figure(figsize=figsize)
    
    for i, col in enumerate(categorical_cols):
        # Get value counts and calculate percentages
        print(f"Number of categories for {col}", len(df[col].unique()))
        # Log number of categories
        mlflow.log_param(f"{col.lower()}_unique_categories", len(df[col].unique()))
        
        value_counts = df[col].value_counts().head(10)  # Top 10 categories
        total = len(df[~df[col].isna()])
        percentages = (value_counts / total * 100).round(1)
        
        # Add count plot
        ax1 = fig.add_subplot(n_rows, n_cols, i*2+1)
        bars = ax1.bar(
            value_counts.index, 
            value_counts.values, 
            color='skyblue'
        )
        ax1.set_title(f'Distribution of {col}', fontsize=12)
        ax1.set_ylabel('Count')
        
        # Add percentage labels on bars
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{percentage}%',
                ha='center', 
                va='bottom',
                fontsize=8
            )
        
        # Rotate x-axis labels if needed
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        
        # Add price relationship plot
        ax2 = fig.add_subplot(n_rows, n_cols, i*2+2)
        
        # For variables with few categories, use boxplot
        if df[col].nunique() <= 10:
            sns.boxplot(x=col, y='Price', data=df, ax=ax2)
            ax2.set_title(f'Price by {col}', fontsize=12)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            
            # Log average price by category
            for category, avg_price in df.groupby(col)['Price'].mean().items():
                mlflow.log_metric(f"avg_price_by_{col.lower()}_{category}", avg_price)
        # For variables with many categories, use bar plot of average prices
        else:
            # Calculate average price by category
            avg_price = df.groupby(col)['Price'].mean().sort_values(ascending=False).head(10)
            ax2.bar(avg_price.index, avg_price.values, color='lightgreen')
            ax2.set_title(f'Avg Price by {col} (Top 10)', fontsize=12)
            ax2.set_ylabel('Average Price ($)')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            
            # Format y-axis as currency
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f'${int(x/1000)}K'))
        
    
    plt.tight_layout()
    
    # Save and log the figure
    fig_path = "categorical_distributions.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)
    
    plt.show()

categorical_cols = ['Type', 'Method', 'Seller', 'CouncilArea', 'Region']
plot_categorical_distributions(df, categorical_cols)

plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 2)
top_suburbs_by_price = df.groupby('Suburb')['Price'].mean().sort_values(ascending=False).head(15)
sns.barplot(x=top_suburbs_by_price.index, y=top_suburbs_by_price.values)
plt.title('Top 15 Suburbs by Average Price', fontsize=14)
plt.ylabel('Average Price ($)')
plt.xticks(rotation=45, ha='right')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f'${int(x/1000)}K'))

plt.tight_layout()

# Save and log the figure
fig_path = "top_suburbs_by_price.png"
plt.savefig(fig_path)
mlflow.log_artifact(fig_path)

plt.show()

# Log top 5 suburbs by price
for suburb, price in top_suburbs_by_price.head(5).items():
    mlflow.log_metric(f"top_suburb_price_{suburb}", price)



# 2. Check for logical constraints on bathrooms and bedrooms
print("\nProperties with zero bathrooms:")
zero_bathrooms = df[df['Bathroom'] == 0]
print(f"Count: {len(zero_bathrooms)} ({len(zero_bathrooms)/len(df)*100:.2f}%)")
display(zero_bathrooms[["Rooms", "Type", "Price", "Bedroom2", "Bathroom", "Landsize", "BuildingArea"]].head(10))
mlflow.log_text(zero_bathrooms[["Rooms", "Type", "Price", "Bedroom2", "Bathroom", "Landsize", "BuildingArea"]].head(10).to_string(),
                "zero_bathrooms_sample.txt")


print("\nProperties with zero bedrooms:")
zero_bedrooms = df[df['Bedroom2'] == 0]
print(f"Count: {len(zero_bedrooms)} ({len(zero_bedrooms)/len(df)*100:.2f}%)")
display(zero_bedrooms[["Rooms", "Type", "Price", "Bedroom2", "Bathroom", "Landsize", "BuildingArea"]].head(10))
mlflow.log_text(zero_bedrooms[["Rooms", "Type", "Price", "Bedroom2", "Bathroom", "Landsize", "BuildingArea"]].head(10).to_string(), 
                "zero_bedrooms_sample.txt")



# 1. Rooms vs. Bedroom2 + Bathroom consistency
print("1. Rooms vs. Bedroom2 + Bathroom:")
rooms_vs_components = df[['Rooms', 'Bedroom2', 'Bathroom']].dropna()
# Calculate total components (bedrooms + bathrooms)
rooms_vs_components['TotalComponents'] = rooms_vs_components['Bedroom2'] + rooms_vs_components['Bathroom']
# Find inconsistent records where components exceed total rooms
inconsistent_components = rooms_vs_components[rooms_vs_components['Rooms'] < rooms_vs_components['TotalComponents']]
print(f"Properties with more bedrooms+bathrooms than total rooms: {len(inconsistent_components)} ({len(inconsistent_components)/len(rooms_vs_components)*100:.2f}%)")

if len(inconsistent_components) > 0:
    print("\nSample of inconsistent records (Rooms < Bedroom2 + Bathroom):")
    display(inconsistent_components.head(5))
    mlflow.log_text(inconsistent_components.head(5).to_string(), "inconsistent_rooms_sample.txt")

# 3. Check for properties with unrealistic YearBuilt values
print("\nProperties with unrealistic YearBuilt values:")
unrealistic_year = df[(df['YearBuilt'] < 1840) & (~df['YearBuilt'].isna())]
print(f"Count: {len(unrealistic_year)} ({len(unrealistic_year)/len(df[~df['YearBuilt'].isna()])*100:.2f}%)")

if len(unrealistic_year) > 0:
    print("\nSample of properties with unrealistic YearBuilt values:")
    display(unrealistic_year[['Address', 'YearBuilt', 'Price']].head(5))
    mlflow.log_text(unrealistic_year[['Address', 'YearBuilt', 'Price']].head(5).to_string(), 
                    "unrealistic_year_sample.txt")


# Check for properties built in the future (after data collection)
# Assuming data collection ended in 2018 (based on max date in dataset)
future_buildings = df[(df['YearBuilt'] > 2018) & (~df['YearBuilt'].isna())]
print(f"Properties with future YearBuilt values (>2018): {len(future_buildings)} ({len(future_buildings)/len(df[~df['YearBuilt'].isna()])*100:.2f}%)")


# 4. Check for properties with zero land size but non-zero building area
print("\nProperties with zero land size but non-zero building area:")
inconsistent_area = df[(df['Landsize'] == 0) & (df['BuildingArea'] > 0) & (~df['BuildingArea'].isna())]
print(f"Count: {len(inconsistent_area)} ({len(inconsistent_area)/len(df)*100:.2f}% of dataset)")
display(inconsistent_area[["Rooms", "Type", "Price", "Bedroom2", "Bathroom", "Landsize", "BuildingArea"]].head(10))
mlflow.log_text(inconsistent_area[["Rooms", "Type", "Price", "Bedroom2", "Bathroom", "Landsize", "BuildingArea"]].head(10).to_string(), 
                "inconsistent_area_sample.txt")


# Count by property type
type_counts = inconsistent_area['Type'].value_counts()

# Create pie chart
plt.figure(figsize=(10, 7))
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=90, shadow=True)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Properties with Zero Land Size but Non-Zero Building Area by Type')
plt.tight_layout()
# Save figure to buffer and log as artifact
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
mlflow.log_image(Image.open(buf), "inconsistent_area_by_type.png")

plt.show()


# Calculate percentages within each property type
property_types = df['Type'].unique()
print("\nPercentage of each property type with this inconsistency:")
for prop_type in property_types:
    total_of_type = len(df[df['Type'] == prop_type])
    inconsistent_of_type = len(inconsistent_area[inconsistent_area['Type'] == prop_type])
    if total_of_type > 0:
        percentage = (inconsistent_of_type / total_of_type) * 100
        mlflow.log_metric(f"inconsistent_percentage_type_{prop_type}", percentage)
        print(f"Type '{prop_type}': {inconsistent_of_type}/{total_of_type} ({percentage:.2f}% of {prop_type})")


# Check for units (Type 'u') with non-zero land size
units_with_land = df[(df['Type'] == 'u') & (df['Landsize'] > 0) & (~df['Landsize'].isna())]
print(f"Units with non-zero land size: {len(units_with_land)} ({len(units_with_land)/len(df[df['Type'] == 'u'])*100:.2f}% of units)")

# Display sample of these units with their land sizes
print("\nSample of units with non-zero land size:")
display(units_with_land[['Address', 'Type', 'Landsize', "BuildingArea", 'Price']].head(10))
mlflow.log_text(units_with_land[['Address', 'Type', 'Landsize', "BuildingArea", 'Price']].head(10).to_string(), 
                "units_with_land_sample.txt")

# Calculate building area to land size ratio
building_exceeds_land = df[(df['BuildingArea'] > df['Landsize']) & 
                          (df['Landsize'] > 0) & 
                          (~df['BuildingArea'].isna())].copy()

# Add ratio column
building_exceeds_land['BuildingToLandRatio'] = building_exceeds_land['BuildingArea'] / building_exceeds_land['Landsize']

# Create categories based on ratio
building_exceeds_land['RatioCategory'] = pd.cut(
    building_exceeds_land['BuildingToLandRatio'],
    bins=[1, 2, 3, float('inf')],
    labels=['1-2x', '2-3x', '>3x']
)

# Count by ratio category and property type
ratio_type_counts = pd.crosstab(building_exceeds_land['Type'], building_exceeds_land['RatioCategory'])
print("Distribution of building-to-land ratios by property type:")
display(ratio_type_counts)
mlflow.log_text(ratio_type_counts.to_string(), "ratio_type_distribution.txt")

# Calculate percentages
ratio_percentages = ratio_type_counts.div(ratio_type_counts.sum(axis=1), axis=0) * 100
print("\nPercentage distribution of ratios within each property type:")
display(ratio_percentages.round(2))
mlflow.log_text(ratio_percentages.round(2).to_string(), "ratio_percentages_by_type.txt")


# Create pie charts for each ratio category
plt.figure(figsize=(18, 6))

# Plot for ratios 1-2x
plt.subplot(1, 3, 1)
ratio_1_2 = building_exceeds_land[building_exceeds_land['RatioCategory'] == '1-2x']['Type'].value_counts()
plt.pie(ratio_1_2, labels=ratio_1_2.index, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Properties with Building Area 1-2x Land Size')

# Plot for ratios 2-3x
plt.subplot(1, 3, 2)
ratio_2_3 = building_exceeds_land[building_exceeds_land['RatioCategory'] == '2-3x']['Type'].value_counts()
plt.pie(ratio_2_3, labels=ratio_2_3.index, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Properties with Building Area 2-3x Land Size')

# Plot for ratios >3x
plt.subplot(1, 3, 3)
ratio_3_plus = building_exceeds_land[building_exceeds_land['RatioCategory'] == '>3x']['Type'].value_counts()
plt.pie(ratio_3_plus, labels=ratio_3_plus.index, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Properties with Building Area >3x Land Size')

plt.tight_layout()
# Save figure to buffer and log as artifact
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
mlflow.log_image(Image.open(buf), "building_to_land_ratio_by_type.png")
plt.show()


# Show extreme cases (highest ratios)
print("\nTop 10 properties with highest building-to-land ratios:")
display(building_exceeds_land[['Address', 'Type', 'Landsize', 'BuildingArea', 'BuildingToLandRatio']]
        .sort_values('BuildingToLandRatio', ascending=False)
        .head(10))
mlflow.log_text(building_exceeds_land[['Address', 'Type', 'Landsize', 'BuildingArea', 'BuildingToLandRatio']]
        .sort_values('BuildingToLandRatio', ascending=False)
        .head(10).to_string(), "extreme_ratio_properties.txt")

# Summary statistics of ratios by property type
print("\nSummary statistics of building-to-land ratios by property type:")
display(building_exceeds_land.groupby('Type')['BuildingToLandRatio'].describe())

# 9. Visualize the relationship between BuildingArea and Landsize
valid_areas = df[(df['BuildingArea'] > 0) & (df['Landsize'] > 0) & (~df['BuildingArea'].isna())]
plt.figure(figsize=(10, 6))
plt.scatter(valid_areas['Landsize'], valid_areas['BuildingArea'], alpha=0.3)
plt.plot([0, 2000], [0, 2000], 'r--')
plt.xlabel('Land Size (sqm)')
plt.ylabel('Building Area (sqm)')
plt.title('Consistency Check: Land Size vs Building Area')
plt.xlim(0, 2000)
plt.ylim(0, 2000)
plt.grid(True, alpha=0.3)
plt.annotate('Consistentency Line\n(Anything above: Building Area > Land Size)', 
             xy=(500, 500), 
             xytext=(700, 1500),
             arrowprops=dict(facecolor='red', shrink=0.05))
plt.tight_layout()
# Save figure to buffer and log as artifact
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
mlflow.log_image(Image.open(buf), "land_vs_building_area.png")

plt.show()


print("\n3. Price vs. BuildingArea consistency:")
# Calculate price per square meter
df['PricePerSqm'] = df['Price'] / df['BuildingArea']
# Define reasonable bounds for Melbourne property market
extreme_price_per_sqm = df[(~df['PricePerSqm'].isna()) & 
                          ((df['PricePerSqm'] < 1000) | (df['PricePerSqm'] > 20000))]
print(f"Properties with extreme price per square meter (<$1000 or >$20000): {len(extreme_price_per_sqm)} ({len(extreme_price_per_sqm)/len(df[~df['PricePerSqm'].isna()])*100:.2f}%)")

if len(extreme_price_per_sqm) > 0:
    print("\nSample of properties with extreme price per square meter:")
    display(extreme_price_per_sqm[['Address', 'Type', 'Price', 'BuildingArea', 'PricePerSqm']].head(5))
    mlflow.log_text(
        extreme_price_per_sqm[['Address', 'Type', 'Price', 'BuildingArea', 'PricePerSqm']].head(5).to_string(),
        "extreme_price_per_sqm_sample.txt"
    )

df = df.drop('PricePerSqm', axis=True)

# Create a summary bar plot using existing variables from the notebook

# Collect issue counts from existing variables
issues = {
    'Zero Land Size with Building': len(inconsistent_area),
    'Building Area > Land Size': len(building_exceeds_land),
    'Extreme Price per sqm': len(extreme_price_per_sqm),
    'Zero Bathrooms': len(zero_bathrooms),
    'Zero Bedrooms': len(zero_bedrooms),
    'Unrealistic YearBuilt': len(unrealistic_year)
}
for issue_name, issue_count in issues.items():
    mlflow.log_metric(f"issue_count_{issue_name.replace('>', '').lower()}", issue_count)
    
# Sort by count
sorted_issues = dict(sorted(issues.items(), key=lambda item: item[1], reverse=True))

# Create the bar plot
plt.figure(figsize=(12, 6), dpi = 300)
bars = plt.bar(sorted_issues.keys(), sorted_issues.values(), color='skyblue')
plt.title('Data Consistency Issues in Melbourne Housing Dataset', fontsize=14)
plt.ylabel('Number of Properties', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', alpha=0.3)

# Add count labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{int(height):,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
mlflow.log_image(Image.open(buf), "data_consistency_issues_summary.png")

plt.show()


# Select numerical columns for correlation analysis
numerical_cols = [
    'Rooms', 'Price', 'Distance', 'Bedroom2', 'Bathroom',
    'Car', 'Landsize', 'BuildingArea', 'YearBuilt',
    'Propertycount', 'Lattitude', 'Longtitude'
]  # Note that Latitude and Longitude is misspelled.

# Create correlation matrix
correlation_matrix = df[numerical_cols].corr()
display(correlation_matrix)
mlflow.log_text(correlation_matrix.to_string(), "correlation_matrix.txt")


# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,           # Show correlation values
    cmap='coolwarm',      # Color scheme
    fmt='.2f',            # Format for correlation values
    linewidths=0.5,       # Width of grid lines
    vmin=-1, vmax=1       # Value range
)
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
mlflow.log_image(Image.open(buf), "correlation_heatmap.png")
plt.show()


# Calculate correlation specifically with Price
price_correlation = correlation_matrix['Price'].sort_values(ascending=False)
print("Correlation with Price:")
print(price_correlation)
mlflow.log_text(price_correlation.to_string(), "price_correlations.txt")


# Create a horizontal bar chart of correlations with Price
plt.figure(figsize=(10, 8))
price_correlation.drop('Price').plot(kind='barh', color=plt.cm.coolwarm(price_correlation.drop('Price')/2 + 0.5))
plt.title('Correlation with Property Price', fontsize=16)
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
mlflow.log_image(Image.open(buf), "price_correlation_barplot.png")
plt.show()


# Identify categorical columns
categorical_cols = ['Suburb', 'Type', 'Method', 'Seller', 'CouncilArea', 'Region']

# Create a function to calculate Cramer's V statistic for categorical variables
def cramers_v(x, y):
    """Calculate Cramer's V statistic between two categorical variables"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Create a matrix to store Cramer's V values
cramer_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)

# Calculate Cramer's V for each pair of categorical variables
for i in categorical_cols:
    for j in categorical_cols:
        if i == j:
            cramer_matrix.loc[i, j] = 1.0
        else:
            # Handle missing values by using dropna
            valid_data = df[[i, j]].dropna()
            if len(valid_data) > 0:
                cramer_matrix.loc[i, j] = cramers_v(valid_data[i], valid_data[j])
            else:
                cramer_matrix.loc[i, j] = np.nan
display(cramer_matrix.astype(float))
mlflow.log_text(cramer_matrix.astype(float).to_string(), "cramers_v_matrix.txt")


# Create a heatmap for categorical correlations
plt.figure(figsize=(12, 10))
sns.heatmap(
    cramer_matrix.astype(float),
    annot=True,
    cmap="viridis",
    fmt=".2f",
    linewidths=0.5,
    vmin=0,
    vmax=1
)
plt.title("Cramer's V Correlation Matrix for Categorical Features", fontsize=16)
plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
mlflow.log_image(Image.open(buf), "cramers_v_heatmap.png")
plt.show()


# Calculate mutual information between categorical features and Price
def calculate_mi_score(X, y):
    """Calculate mutual information score for features"""
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    return mi_scores.sort_values(ascending=False)

# Prepare categorical data for mutual information calculation
# We need to encode categorical variables
categorical_encoded = pd.DataFrame()
for col in categorical_cols:
    if df[col].dtype == 'object':
        # Skip columns with too many missing values
        if df[col].isna().sum() / len(df) < 0.3:  # Less than 30% missing
            le = LabelEncoder()
            categorical_encoded[col] = le.fit_transform(df[col].fillna('MISSING'))

# Calculate mutual information scores
if not categorical_encoded.empty:
    mi_scores = calculate_mi_score(categorical_encoded, df['Price'].values)
    display(mi_scores)
    mlflow.log_text(mi_scores.to_string(), "mutual_information_scores.txt")


if not categorical_encoded.empty:
    # Plot mutual information scores
    plt.figure(figsize=(10, 6))
    mi_scores.plot(kind='barh', color='teal')
    plt.title('Mutual Information Scores for Categorical Features (Price Prediction)', fontsize=16)
    plt.xlabel('Mutual Information Score', fontsize=12)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    mlflow.log_image(Image.open(buf), "mutual_information_barplot.png")
    plt.show()

transformers = {}

def register_transformer(name, transform_function, fit_function=None):
    """
    Create a FunctionTransformer and register it in the transformers dictionary
    
    Parameters:
    name (str): Name of the transformer
    transform_function (callable): Function to transform the data
    fit_function (callable, optional): Function to fit the transformer
    
    Returns:
    FunctionTransformer: The created transformer
    """
    if fit_function:
        transformer = FunctionTransformer(transform_function, check_inverse=False)
        transformer.fit = fit_function.__get__(transformer)
    else:
        transformer = FunctionTransformer(transform_function, check_inverse=False)
    
    transformers[name] = transformer
    return transformer

def create_complete_pipeline():
    """Create a complete preprocessing pipeline from all registered transformers"""
    steps = []
    
    for name, transformer in transformers.items():
        steps.append((name, transformer))
    
    return Pipeline(steps)

def log(key, value):
    """Log parameter to MLflow and display"""
    mlflow.log_param(key, value)
    display(f"{key}: {value}")


def remove_columns(X):
    X = X.copy()
    if 'Region' in X.columns:
        X = X.drop('Region', axis=1)
    if 'CouncilArea' in X.columns:
        X = X.drop('CouncilArea', axis=1)
    if 'Bedroom2' in X.columns:
        X = X.drop('Bedroom2', axis=1)
    if 'Propertycount' in X.columns:
        X = X.drop('Propertycount', axis=1)
    return X

remove_transformer = register_transformer('remove_columns', remove_columns)

df = remove_transformer.transform(df)

log("remaining_columns", df.columns.tolist())
log("shape_after_column_removal", df.shape)


# Check if the misspelled columns exist before renaming
def fix_column_names(X):
    X = X.copy()
    rename_dict = {}
    if 'Lattitude' in X.columns:
        rename_dict['Lattitude'] = 'Latitude'
    if 'Longtitude' in X.columns:
        rename_dict['Longtitude'] = 'Longitude'
    
    if rename_dict:
        X = X.rename(columns=rename_dict)
    return X

rename_transformer = register_transformer('fix_column_names', fix_column_names)
log("renamed_columns", "Lattitude->Latitude, Longtitude->Longitude")

# Apply the transformer to our dataframe
df = rename_transformer.transform(df)
print("\nUpdated columns:")
print(df.columns.tolist())


def fix_building_area(X):
    X_copy = X.copy()
    
    # Force print to make sure this function is being called    
    # Count NaNs before imputation
    nans_before = X_copy['BuildingArea'].isna().sum()
    print(f"BuildingArea NaNs before imputation: {nans_before}")
    

    
    # Simple imputation by Type and Rooms
    for type_val in X_copy['Type'].unique():
        for rooms_val in X_copy['Rooms'].unique():
            mask = (X_copy['Type'] == type_val) & (X_copy['Rooms'] == rooms_val)
            
            # Skip if no rows match this combination
            if not mask.any():
                continue
                
            # Get non-NaN values for this group
            valid_values = X_copy.loc[mask & X_copy['BuildingArea'].notna(), 'BuildingArea']
            
            # Calculate median if there are valid values
            if len(valid_values) > 0:
                group_median = valid_values.median()
            else:
                # Try type median if no valid values in this group
                type_values = X_copy.loc[X_copy['Type'] == type_val, 'BuildingArea']
                type_values = type_values[type_values.notna()]
                
                if len(type_values) > 0:
                    group_median = type_values.median()
                else:
                    # Use global median as last resort
                    all_values = X_copy['BuildingArea'][X_copy['BuildingArea'].notna()]
                    if len(all_values) > 0:
                        group_median = all_values.median()
                    else:
                        print("WARNING: No valid BuildingArea values found!")
                        group_median = 0  # Default value if no valid data
            
            # Apply imputation
            nan_mask = mask & X_copy['BuildingArea'].isna()
            if nan_mask.any():
                X_copy.loc[nan_mask, 'BuildingArea'] = group_median
    
    # Final check for any remaining NaNs
    remaining_nans = X_copy['BuildingArea'].isna().sum()
    if remaining_nans > 0:
        
        # Get global median from non-NaN values
        global_median = X_copy.loc[X_copy['BuildingArea'].notna(), 'BuildingArea'].median()
        if pd.isna(global_median):
            print("ERROR: Cannot calculate global median, all values are NaN")
            global_median = 0  # Default value
        
        # Apply global imputation
        X_copy['BuildingArea'] = X_copy['BuildingArea'].fillna(global_median)
    
    # Final verification
    final_nans = X_copy['BuildingArea'].isna().sum()
    
    return X_copy




fix_transformer = register_transformer("fix_building_area", fix_building_area)
manual_df = df.copy()

df = fix_transformer.fit_transform(df)
print(f"NaNs after direct imputation: {df['BuildingArea'].isna().sum()}")

print("\n--- MANUAL PIPELINE APPLICATION ---")
for name, transformer in transformers.items():
    print(f"Applying {name}...")
    manual_df = transformer.fit_transform(manual_df)


def car_imputer(X):
    X_copy = X.copy()
    
    # Count NaNs before imputation
    nans_before = X_copy['Car'].isna().sum()
    print(f"Car NaNs before imputation: {nans_before}")
    
    # Simple imputation by Type and Rooms
    for type_val in X_copy['Type'].unique():
        for rooms_val in X_copy['Rooms'].unique():
            mask = (X_copy['Type'] == type_val) & (X_copy['Rooms'] == rooms_val)
            group_median = X_copy.loc[mask, 'Car'].median()
            
            # If group median is NaN, use type median
            if pd.isna(group_median):
                group_median = X_copy.loc[X_copy['Type'] == type_val, 'Car'].median()
                
                # If type median is NaN, use global median
                if pd.isna(group_median):
                    group_median = X_copy['Car'].median()
            
            # Apply imputation
            nan_mask = mask & X_copy['Car'].isna()
            X_copy.loc[nan_mask, 'Car'] = group_median
    
    # Final global imputation for any remaining NaNs
    global_median = X_copy['Car'].median()
    X_copy['Car'] = X_copy['Car'].fillna(global_median)
    
    # Count NaNs after imputation
    nans_after = X_copy['Car'].isna().sum()
    print(f"Car NaNs after imputation: {nans_after}")
    
    return X_copy

imputer_transformer = register_transformer("car_imputer", car_imputer)

df = imputer_transformer.transform(df)
print(f"NaNs after direct imputation: {df['Car'].isna().sum()}")


def bathroom_imputer(X):
    X_copy = X.copy()
    
    # Count zeros before imputation
    zeros_before = (X_copy['Bathroom'] == 0).sum()
    print(f"Bathroom zeros before imputation: {zeros_before}")
    
    # Simple imputation by Type
    bathroom_medians_by_type = X_copy.groupby('Type')['Bathroom'].median()
    
    for property_type in X_copy['Type'].unique():
        mask = (X_copy['Bathroom'] == 0) & (X_copy['Type'] == property_type)
        median_value = bathroom_medians_by_type[property_type]
        
        # If median is 0 or NaN, use global median
        if pd.isna(median_value) or median_value == 0:
            median_value = X_copy[X_copy['Bathroom'] > 0]['Bathroom'].median()
            
        X_copy.loc[mask, 'Bathroom'] = median_value
    
    # Count zeros after imputation
    zeros_after = (X_copy['Bathroom'] == 0).sum()
    print(f"Bathroom zeros after imputation: {zeros_after}")
    
    return X_copy

imputer_transformer = register_transformer("bathroom_imputer", bathroom_imputer)

df = imputer_transformer.transform(df)
print(f"NaNs after direct imputation: {df['Bathroom'].isna().sum()}")


def year_built_imputer(X):
    X_copy = X.copy()
    
    # Count NaNs before imputation
    nans_before = X_copy['YearBuilt'].isna().sum()
    print(f"YearBuilt NaNs before imputation: {nans_before}")
    
    # Simple imputation by Suburb
    for suburb_val in X_copy['Suburb'].unique():
        mask = (X_copy['Suburb'] == suburb_val)
        group_median = X_copy.loc[mask, 'YearBuilt'].median()
        
        # If group median is NaN, use global median
        if pd.isna(group_median):
            group_median = X_copy['YearBuilt'].median()
            
        # Apply imputation
        nan_mask = mask & X_copy['YearBuilt'].isna()
        X_copy.loc[nan_mask, 'YearBuilt'] = group_median
    
    # Final global imputation for any remaining NaNs
    global_median = X_copy['YearBuilt'].median()
    X_copy['YearBuilt'] = X_copy['YearBuilt'].fillna(global_median)
    
    # Count NaNs after imputation
    nans_after = X_copy['YearBuilt'].isna().sum()
    print(f"YearBuilt NaNs after imputation: {nans_after}")
    
    return X_copy

imputer_transformer = register_transformer("year_built_imputer", year_built_imputer)

df = imputer_transformer.transform(df)
print(f"NaNs after direct imputation: {df['YearBuilt'].isna().sum()}")


# Create indicator features for missing values before imputation
def create_missing_indicators(X):
    X = X.copy()
    X['BuildingArea_Missing'] = X['BuildingArea'].isna().astype(int)
    X['YearBuilt_Missing'] = X['YearBuilt'].isna().astype(int)
    return X

register_transformer('create_missing_indicators', create_missing_indicators)

# Apply the transformer to our dataframe
df = create_missing_indicators(df)
log("BuildingArea_missing_count", df['BuildingArea_Missing'].sum())
log("YearBuilt_missing_count", df['YearBuilt_Missing'].sum())


# Log statistics after imputation
missing_after = df[['BuildingArea', 'YearBuilt']].isna().sum()
print("\nRemaining missing values after imputation:")
log("missing_after_imputation", missing_after.to_dict())

# Log correlation of missing indicators with Price
missing_corr = df[['BuildingArea_Missing', 'YearBuilt_Missing', 'Price']].corr()
print("\nCorrelation of missing indicators with Price:")
log("missing_indicators_correlation_with_price", missing_corr['Price'].drop('Price').to_dict())

# Log summary statistics for imputed features
imputed_stats = df[['BuildingArea', 'YearBuilt']].describe().to_dict()
print("\nSummary statistics for imputed features:")
print(df[['BuildingArea', 'YearBuilt']].describe())


# Define transformer for dropping missing indicators
def drop_missing_indicators(X):
    X = X.copy()
    if 'BuildingArea_Missing' in X.columns:
        X = X.drop('BuildingArea_Missing', axis=1)
    if 'YearBuilt_Missing' in X.columns:
        X = X.drop('YearBuilt_Missing', axis=1)
    return X

register_transformer('drop_missing_indicators', drop_missing_indicators)

# Apply the transformer to our dataframe
df = drop_missing_indicators(df)
log("final_shape", df.shape)


# 1. Zero-Value Bathrooms (34 properties)
print(f"Properties with zero bathrooms before imputation: {len(df[df['Bathroom'] == 0])}")

# Calculate median bathrooms by property type
bathroom_medians_by_type = df.groupby('Type')['Bathroom'].median()
print("\nMedian bathroom count by property type:")
print(bathroom_medians_by_type)

# Bathroom imputation transformer
def impute_bathrooms(X):
    X_copy = X.copy()
    bathroom_medians_by_type = X_copy.groupby('Type')['Bathroom'].median()
    
    for property_type in X_copy['Type'].unique():
        mask = (X_copy['Bathroom'] == 0) & (X_copy['Type'] == property_type)
        X_copy.loc[mask, 'Bathroom'] = bathroom_medians_by_type[property_type]
    
    return X_copy

def fit_impute_bathrooms(self, X, y=None):
    self.bathroom_medians_ = X.groupby('Type')['Bathroom'].median()
    return self

bathroom_transformer = register_transformer('bathroom_imputer', impute_bathrooms, fit_impute_bathrooms)

# Apply bathroom imputation
df = bathroom_transformer.fit_transform(df)
log('zero_bathroom_count_after_imputation', len(df[df['Bathroom'] == 0]))


# 2. Unrealistic Construction Dates (pre-1840)
# Identify properties with unrealistic dates
unrealistic_dates = df[df['YearBuilt'] < 1840].copy()
print(f"\nProperties with unrealistic construction dates (pre-1840): {len(unrealistic_dates)}")

if len(unrealistic_dates) > 0:
    print("Properties with unrealistic dates:")
    print(unrealistic_dates[['Address', 'YearBuilt']].to_string())
    
    # Year built correction transformer
    def correct_year_built(X):
        X_copy = X.copy()
        X_copy.loc[X_copy['YearBuilt'] == 1196, 'YearBuilt'] = 1996
        X_copy.loc[X_copy['YearBuilt'] == 1830, 'YearBuilt'] = 1930
        
        if 'PropertyAge' in X_copy.columns:
            current_year = 2023  # Adjust as needed
            X_copy['PropertyAge'] = current_year - X_copy['YearBuilt']
        
        return X_copy
    
    year_built_transformer = register_transformer('year_built_corrector', correct_year_built)
    
    # Apply year built corrections
    df = year_built_transformer.fit_transform(df)
    log('unrealistic_dates_count_after_correction', len(df[df['YearBuilt'] < 1840]))


# 1. Zero Land Size with Building Area (1,061 properties)
zero_land_count = len(df[(df['Landsize'] == 0) & (df['BuildingArea'] > 0)])
print(f"Properties with zero land size but positive building area: {zero_land_count}")
def handle_land_size(X):
    if isinstance(X, pd.DataFrame):
        X_copy = X.copy()
    else:
        # If X is not a DataFrame, convert it
        X_copy = pd.DataFrame(X, columns=df.columns)
    
    # Create the flag column
    X_copy['LandSizeNotOwned'] = ((X_copy['Type'] == 'u') & (X_copy['Landsize'] == 0)).astype(int)
    
    # Rest of the function remains the same
    unit_global_median = X_copy[(X_copy['Type'] == 'u') & (X_copy['Landsize'] > 0)]['Landsize'].median()
    townhouse_global_median = X_copy[(X_copy['Type'] == 't') & (X_copy['Landsize'] > 0)]['Landsize'].median()
    house_global_median = X_copy[(X_copy['Type'] == 'h') & (X_copy['Landsize'] > 0)]['Landsize'].median()
    
    X_copy.loc[(X_copy['Type'] == 'u') & (X_copy['Landsize'] == 0), 'Landsize'] = unit_global_median
    X_copy.loc[(X_copy['Type'] == 't') & (X_copy['Landsize'] == 0), 'Landsize'] = townhouse_global_median
    X_copy.loc[(X_copy['Type'] == 'h') & (X_copy['Landsize'] == 0), 'Landsize'] = house_global_median
    
    other_types = [t for t in X_copy['Type'].unique() if t not in ['u', 't', 'h']]
    if other_types:
        other_global_median = X_copy[(X_copy['Type'].isin(other_types)) & (X_copy['Landsize'] > 0)]['Landsize'].median()
        X_copy.loc[(X_copy['Type'].isin(other_types)) & (X_copy['Landsize'] == 0), 'Landsize'] = other_global_median
    
    return X_copy

def fit_handle_land_size(self, X, y=None):
    self.unit_global_median_ = X[(X['Type'] == 'u') & (X['Landsize'] > 0)]['Landsize'].median()
    self.townhouse_global_median_ = X[(X['Type'] == 't') & (X['Landsize'] > 0)]['Landsize'].median()
    self.house_global_median_ = X[(X['Type'] == 'h') & (X['Landsize'] > 0)]['Landsize'].median()
    
    other_types = [t for t in X['Type'].unique() if t not in ['u', 't', 'h']]
    if other_types:
        self.other_global_median_ = X[(X['Type'].isin(other_types)) & (X['Landsize'] > 0)]['Landsize'].median()
    
    return self

land_size_transformer = register_transformer('land_size_handler', handle_land_size, fit_handle_land_size)

# Apply land size handling
df = land_size_transformer.fit_transform(df)
log('zero_land_count_after_imputation', len(df[df['Landsize'] == 0]))
log('land_size_not_owned_count', df['LandSizeNotOwned'].sum())


# Log mutual information if Price column exists
if 'Price' in df.columns:
    X = df[['LandSizeNotOwned']].values
    y = df['Price'].values
    mi_score = mutual_info_regression(X, y)[0]
    log("LandSizeNotOwned_mutual_info_score", mi_score)


# Building area imputation transformer
def impute_building_area(X):
    X_copy = X.copy()
    
    for property_type in X_copy["Type"].unique():
        for room_count in X_copy["Rooms"].unique():
            mask = (X_copy["BuildingArea"] == 0) & (X_copy["Type"] == property_type) & (X_copy["Rooms"] == room_count)

            median_area = X_copy[(X_copy["BuildingArea"] > 0) & 
                              (X_copy["Type"] == property_type) & 
                              (X_copy["Rooms"] == room_count)]["BuildingArea"].median()
            
            if pd.isna(median_area):
                median_area = X_copy[(X_copy["BuildingArea"] > 0) & 
                                  (X_copy["Type"] == property_type)]["BuildingArea"].median()
                
            if pd.isna(median_area):
                median_area = X_copy[X_copy["BuildingArea"] > 0]["BuildingArea"].median()
            
            X_copy.loc[mask, "BuildingArea"] = median_area
    
    return X_copy

def fit_impute_building_area(self, X, y=None):
    self.building_area_medians_ = {}
    for property_type in X["Type"].unique():
        self.building_area_medians_[property_type] = {}
        for room_count in X["Rooms"].unique():
            median = X[(X["BuildingArea"] > 0) & 
                        (X["Type"] == property_type) & 
                        (X["Rooms"] == room_count)]["BuildingArea"].median()
            self.building_area_medians_[property_type][room_count] = median
    
    self.type_medians_ = X.groupby("Type")["BuildingArea"].median()
    self.global_median_ = X[X["BuildingArea"] > 0]["BuildingArea"].median()
    
    return self

building_area_transformer = register_transformer('building_area_imputer', impute_building_area, fit_impute_building_area)

# Apply building area imputation
df = building_area_transformer.fit_transform(df)
log('zero_building_area_after_imputation', len(df[df["BuildingArea"] == 0]))


def remove_outliers(X):
    X_copy = X.copy()
    
    # Calculate floor area ratio and remove extreme values
    X_copy['FloorAreaRatio'] = X_copy['BuildingArea'] / X_copy['Landsize']
    X_copy = X_copy[X_copy['FloorAreaRatio'] <= 3]
    X_copy = X_copy.drop('FloorAreaRatio', axis=1)
    
    # Remove extreme building areas
    X_copy = X_copy[~(X_copy['BuildingArea'] >= X_copy['BuildingArea'].quantile(0.99))]
    
    # Cap extreme price per square meter values if Price column exists
    if 'Price' in X_copy.columns:
        X_copy['PricePerSqm'] = X_copy['Price'] / X_copy['BuildingArea']
        p01 = X_copy['PricePerSqm'].quantile(0.01)
        p99 = X_copy['PricePerSqm'].quantile(0.99)
        
        X_copy.loc[X_copy['PricePerSqm'] < p01, 'PricePerSqm'] = p01
        X_copy.loc[X_copy['PricePerSqm'] > p99, 'PricePerSqm'] = p99
        
        X_copy = X_copy.drop('PricePerSqm', axis=1)
    
    return X_copy

outlier_transformer = register_transformer('outlier_remover', remove_outliers)

# Apply outlier removal
rows_before_outlier_removal = len(df)
df = outlier_transformer.fit_transform(df)
log('rows_removed_by_outlier_filtering', rows_before_outlier_removal - len(df))


# 1. Property Age Transformer
def transform_property_age(X):
    """Create property age feature from YearBuilt"""
    X_copy = X.copy()
    if 'YearBuilt' in X_copy.columns:
        current_year = 2018
        X_copy['PropertyAge'] = current_year - X_copy['YearBuilt']
        X_copy = X_copy.drop('YearBuilt', axis=1)
    return X_copy

property_age_transformer = register_transformer('property_age', transform_property_age)

# Apply and log results
df = property_age_transformer.transform(df)
log('property_age_transformer', f"Created PropertyAge feature, shape: {df.shape}")


# Define the coordinates we're using for Melbourne CBD
CBD_LATITUDE = -37.8136
CBD_LONGITUDE = 144.9631

print(f"Currently using coordinates for Melbourne CBD: ({CBD_LATITUDE}, {CBD_LONGITUDE})")

# 1. Check if these coordinates fall within our dataset
# First, ensure we have the correct column names
lat_col = 'Latitude' if 'Latitude' in df.columns else 'Lattitude'
long_col = 'Longitude' if 'Longitude' in df.columns else 'Longtitude'

# Get the bounds of our data
lat_min, lat_max = df[lat_col].min(), df[lat_col].max()
long_min, long_max = df[long_col].min(), df[long_col].max()

print(f"\nDataset geographical bounds:")
print(f"Latitude: {lat_min} to {lat_max}")
print(f"Longitude: {long_min} to {long_max}")

# Check if CBD coordinates fall within these bounds
if lat_min <= CBD_LATITUDE <= lat_max and long_min <= CBD_LONGITUDE <= long_max:
    print("✓ CBD coordinates fall within the dataset's geographical bounds")
else:
    print("⚠ CBD coordinates fall outside the dataset's geographical bounds")

# 2. Visualize the data points and CBD location
plt.figure(figsize=(10, 8))
plt.scatter(df[long_col], df[lat_col], alpha=0.1, s=1, c='blue', label='Properties')
plt.scatter(CBD_LONGITUDE, CBD_LATITUDE, color='red', s=100, marker='*', label='CBD (Current)')

# 3. Find properties closest to the CBD
df['DistanceToCBD_direct'] = np.sqrt(
    (df[lat_col] - CBD_LATITUDE)**2 + 
    (df[long_col] - CBD_LONGITUDE)**2
)

closest_properties = df.nsmallest(5, 'DistanceToCBD_direct')
print("\nProperties closest to our CBD coordinates:")
print(closest_properties[[lat_col, long_col, 'Suburb', 'Address', 'DistanceToCBD_direct']].to_string())

# Plot these closest properties
plt.scatter(
    closest_properties[long_col], 
    closest_properties[lat_col], 
    color='green', 
    s=50, 
    marker='o', 
    label='Closest Properties'
)

# 4. Check if the 'Distance' column in the dataset is consistent with our coordinates
if 'Distance' in df.columns:
    # Calculate correlation between our calculated distance and the dataset's Distance column
    correlation = df['DistanceToCBD_direct'].corr(df['Distance'])
    print(f"\nCorrelation between calculated distance and dataset's Distance column: {correlation:.4f}")
    
    # If correlation is high, our coordinates are likely correct
    if correlation > 0.9:
        print("✓ High correlation suggests our CBD coordinates are accurate")
    else:
        print("⚠ Low correlation suggests our CBD coordinates may need adjustment")
    
    # Find the coordinates that would minimize the difference with the Distance column
    # This is a simplified approach - for a more accurate result, we would use optimization
    min_diff_idx = (df['DistanceToCBD_direct'] - df['Distance']).abs().idxmin()
    potential_cbd_lat = df.loc[min_diff_idx, lat_col]
    potential_cbd_long = df.loc[min_diff_idx, long_col]
    
    print(f"\nPotential alternative CBD coordinates based on Distance column:")
    print(f"Latitude: {potential_cbd_lat}, Longitude: {potential_cbd_long}")
    
    # Plot this alternative CBD point
    plt.scatter(
        potential_cbd_long, 
        potential_cbd_lat, 
        color='orange', 
        s=100, 
        marker='*', 
        label='Alternative CBD'
    )

# 5. Look up known landmarks in Melbourne CBD
# Flinders Street Station is a well-known central point in Melbourne CBD
FLINDERS_ST_LAT = -37.8183
FLINDERS_ST_LONG = 144.9671

# Plot Flinders Street Station
plt.scatter(
    FLINDERS_ST_LONG, 
    FLINDERS_ST_LAT, 
    color='purple', 
    s=100, 
    marker='*', 
    label='Flinders St Station'
)

print(f"\nFlinders Street Station coordinates: ({FLINDERS_ST_LAT}, {FLINDERS_ST_LONG})")
print(f"Distance between our CBD and Flinders St: {np.sqrt((CBD_LATITUDE-FLINDERS_ST_LAT)**2 + (CBD_LONGITUDE-FLINDERS_ST_LONG)**2):.6f} degrees")

# Finalize the plot
plt.title('Melbourne Properties and CBD Location Verification')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Based on all verification steps, decide on final CBD coordinates
print("\nBased on verification:")
if 'Distance' in df.columns and correlation > 0.9:
    print("Using original CBD coordinates: ({}, {})".format(CBD_LATITUDE, CBD_LONGITUDE))
    final_cbd_lat, final_cbd_long = CBD_LATITUDE, CBD_LONGITUDE
else:
    # Choose the better coordinates based on our analysis
    # This could be Flinders St Station or the alternative derived from Distance
    print("Adjusting to more accurate CBD coordinates: ({}, {})".format(FLINDERS_ST_LAT, FLINDERS_ST_LONG))
    final_cbd_lat, final_cbd_long = FLINDERS_ST_LAT, FLINDERS_ST_LONG

# Return the verified coordinates for use in the direction calculation
CBD_LATITUDE = final_cbd_lat
CBD_LONGITUDE = final_cbd_long
print(f"Final CBD coordinates for direction calculation: ({CBD_LATITUDE}, {CBD_LONGITUDE})")

df = df.drop("DistanceToCBD_direct", axis=1)


def transform_cbd_distance(X):
    """Calculate distance and direction from CBD"""
    X_copy = X.copy()
    
    # Define CBD coordinates
    CBD_LATITUDE = -37.8183  # Using Flinders St coordinates as verified in the code
    CBD_LONGITUDE = 144.9671
    
    lat_col = 'Latitude' if 'Latitude' in X_copy.columns else 'Lattitude'
    long_col = 'Longitude' if 'Longitude' in X_copy.columns else 'Longtitude'
    
    if lat_col in X_copy.columns and long_col in X_copy.columns:
        # Calculate distance to CBD
        X_copy['DistanceToCBD'] = np.sqrt(
            (X_copy[lat_col] - CBD_LATITUDE)**2 + 
            (X_copy[long_col] - CBD_LONGITUDE)**2
        )
        
        # Calculate angle and direction from CBD
        X_copy['AngleFromCBD'] = np.degrees(np.arctan2(
            X_copy[lat_col] - CBD_LATITUDE,
            X_copy[long_col] - CBD_LONGITUDE
        ))
        
        # Convert angle to cardinal direction
        def get_direction(angle):
            angle = angle % 360
            if angle >= 315 or angle < 45:
                return 'E'  
            elif angle >= 45 and angle < 135:
                return 'N' 
            elif angle >= 135 and angle < 225:
                return 'W'
            else:  
                return 'S'
        
        X_copy['DirectionFromCBD'] = X_copy['AngleFromCBD'].apply(get_direction)
        
        # Create direction dummies
        direction_dummies = pd.get_dummies(X_copy['DirectionFromCBD'], prefix='Direction')
        X_copy = pd.concat([X_copy, direction_dummies], axis=1)
    
    return X_copy

cbd_transformer = register_transformer('cbd_features', transform_cbd_distance)

# Apply and log results
df = cbd_transformer.transform(df)
log('cbd_transformer', f"Created CBD distance and direction features, shape: {df.shape}")


# Count properties in each direction
direction_counts = df['DirectionFromCBD'].value_counts()
print("\nProperties by direction from CBD:")
print(direction_counts)

# Calculate average price by direction
if 'Price' in df.columns:
    avg_price_by_direction = df.groupby('DirectionFromCBD')['Price'].mean().sort_values(ascending=False)
    print("\nAverage price by direction from CBD:")
    for direction, avg_price in avg_price_by_direction.items():
        print(f"{direction}: ${avg_price:,.2f}")

# Calculate median price by direction (often more representative than mean)
if 'Price' in df.columns:
    median_price_by_direction = df.groupby('DirectionFromCBD')['Price'].median().sort_values(ascending=False)
    print("\nMedian price by direction from CBD:")
    for direction, median_price in median_price_by_direction.items():
        print(f"{direction}: ${median_price:,.2f}")


# Display the new direction features
print("\nNew direction features added:")
direction_columns = ['DirectionFromCBD'] + [col for col in df.columns if col.startswith('Direction_')]
print(direction_columns)

# Verify columns were removed
print("\nVerifying latitude and longitude columns were removed:")
remaining_geo_cols = [col for col in df.columns if col in [lat_col, long_col]]
if remaining_geo_cols:
    print(f"Warning: {remaining_geo_cols} still present in dataframe")
else:
    print("Confirmed: Latitude and longitude columns successfully removed")

# Calculate mutual information with Price for DirectionFromCBD
if 'Price' in df.columns:
    
    # Prepare the data
    X = pd.get_dummies(df['DirectionFromCBD'])
    y = df['Price'].values
    
    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y)
    
    # Create a DataFrame to display the results
    mi_results = pd.DataFrame({
        'Direction': X.columns,
        'Mutual Information': mi_scores
    })
    
    print("\nMutual Information between Direction and Price:")
    print(mi_results.sort_values('Mutual Information', ascending=False))


print("\nInterpreting Direction from CBD Mutual Information Results:")
print("----------------------------------------------------------")
print("Northern properties show the strongest relationship with price (MI: 0.028)")
print("Western and Eastern properties have similar moderate relationships (MI: ~0.025)")
print("Southern properties show the weakest relationship with price (MI: 0.003)")

# Let's further analyze the price distribution by direction

# Create a boxplot of prices by direction
plt.figure(figsize=(12, 6))
sns.boxplot(x='DirectionFromCBD', y='Price', data=df, order=['N', 'E', 'W', 'S'])
plt.title('Price Distribution by Direction from CBD')
plt.xlabel('Direction from CBD')
plt.ylabel('Price ($)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()



def drop_intermediate_cols(X):
    X_copy = X.copy()
    X_copy = X_copy.drop(['AngleFromCBD', 'DirectionFromCBD', lat_col, long_col], axis=1)
    return X_copy

drop_cols_transformer = register_transformer('drop_intermediate_cols', drop_intermediate_cols)

# Apply and log results
df = drop_cols_transformer.transform(df)
log('drop_cols_transformer', f"Droped intermediate cols, shape: {df.shape}")



# 1. Property Type: One-hot encoding with logical groupings
print("\n1. Property Type encoding")
print(f"Original property types: {df['Type'].unique()}")

def transform_property_type(X):
    """Encode property types"""
    X_copy = X.copy()
    
    if 'Type' in X_copy.columns:
        def group_property_types(prop_type):
            if prop_type == 'h':
                return 'House'
            elif prop_type == 'u':
                return 'Unit/Apartment'
            elif prop_type == 't':
                return 'Townhouse'
            return prop_type
        
        X_copy['PropertyType'] = X_copy['Type'].apply(group_property_types)
        property_type_dummies = pd.get_dummies(X_copy['PropertyType'], prefix='PropType')
        X_copy = pd.concat([X_copy, property_type_dummies], axis=1)
        X_copy = X_copy.drop(['Type', 'PropertyType'], axis=1)
    
    return X_copy

property_type_transformer = register_transformer('property_type', transform_property_type)

# Apply and log results
df = property_type_transformer.transform(df)
log('property_type_transformer', f"Encoded property types, shape: {df.shape}")
print(f"Created property type features: {[col for col in df.columns if col.startswith('PropType_')]}")


# 2. Method of Sale: One-hot encoding
print("\n2. Method of Sale encoding")
print(f"Original sale methods: {df['Method'].unique()}")

# Method of Sale Transformer
def transform_method(X):
    """Encode method of sale"""
    X_copy = X.copy()
    
    if 'Method' in X_copy.columns:
        method_dummies = pd.get_dummies(X_copy['Method'], prefix='Method')
        X_copy = pd.concat([X_copy, method_dummies], axis=1)
        X_copy = X_copy.drop('Method', axis=1)
    
    return X_copy

method_transformer = register_transformer('method', transform_method)

# Apply and log results
df = method_transformer.transform(df)
log('method_transformer', f"Encoded method of sale, shape: {df.shape}")


print(f"Created method features: {[col for col in df.columns if col.startswith('Method_')]}")


suburb_price_ranks = {}

class SuburbTargetEncoder:
    def __init__(self):
        self.suburb_price_rank = None
        self.suburb_to_rank_dict = {}  # Add a dictionary for lookup

    def fit(self, X, y=None):
        if 'Suburb' in X.columns and 'Price' in X.columns:
            suburb_avg_price = X.groupby('Suburb')['Price'].mean()
            self.suburb_price_rank = suburb_avg_price.rank(pct=True)
            
            # Create the lookup dictionary
            self.suburb_to_rank_dict = self.suburb_price_rank.to_dict()
            
            suburb_price_ranks= self.suburb_to_rank_dict
            
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if 'Suburb' in X_copy.columns and self.suburb_price_rank is not None:
            X_copy['Suburb_PriceRank'] = X_copy['Suburb'].map(self.suburb_price_rank)
            X_copy = X_copy.drop('Suburb', axis=1)
        
        return X_copy

def fit_suburb_encoder(self, X, y=None):
    self.encoder = SuburbTargetEncoder()
    self.encoder.fit(X, y)
    return self

def transform_suburb(X):
    if not hasattr(transform_suburb, 'encoder'):
        transform_suburb.encoder = SuburbTargetEncoder()
        if 'Suburb' in X.columns and 'Price' in X.columns:
            transform_suburb.encoder.fit(X)

    return transform_suburb.encoder.transform(X)

suburb_transformer = register_transformer('suburb', transform_suburb, fit_suburb_encoder)
df = suburb_transformer.fit(df).transform(df)
log('suburb_transformer', f"Target encoded suburbs, shape: {df.shape}")


print("\n4. Seller encoding")
print(f"Number of unique sellers: {df['Seller'].nunique()}")


major_sellers = []
def transform_seller(X):
    """Encode sellers, grouping less frequent ones"""
    X_copy = X.copy()

    if 'Seller' in X_copy.columns:
        seller_counts = X_copy['Seller'].value_counts()
        major_sellers = seller_counts[seller_counts >= 100].index.tolist()
    
        
        X_copy['SellerGroup'] = X_copy['Seller'].apply(lambda x: x if x in major_sellers else 'Other')
        seller_dummies = pd.get_dummies(X_copy['SellerGroup'], prefix='Seller')
        X_copy = pd.concat([X_copy, seller_dummies], axis=1)
        X_copy = X_copy.drop(['Seller', 'SellerGroup'], axis=1)
    
    return X_copy

seller_transformer = register_transformer('seller', transform_seller)
df = seller_transformer.transform(df)
log('seller_transformer', f"Encoded sellers, shape: {df.shape}")
print(f"Created seller features: {len([col for col in df.columns if col.startswith('Seller_')])} columns")



# Calculate mutual information for the new features
if 'Price' in df.columns:
    
    # Prepare feature groups for MI calculation
    feature_groups = {
        'Property Type': [col for col in df.columns if col.startswith('PropType_')],
        'Method': [col for col in df.columns if col.startswith('Method_')],
        'Suburb': ['Suburb_PriceRank'],
        'Seller': [col for col in df.columns if col.startswith('Seller_')]
    }
    
    print("\nMutual Information with Price for new encoded features:")
    for group_name, features in feature_groups.items():
        if features:
            X = df[features].values
            y = df['Price'].values
            
            # Handle single-feature case
            if len(features) == 1:
                X = X.reshape(-1, 1)
            
            mi_scores = mutual_info_regression(X, y)
            
            # Create a DataFrame to display the results
            mi_results = pd.DataFrame({
                'Feature': features,
                'Mutual Information': mi_scores
            }).sort_values('Mutual Information', ascending=False)
            
            print(f"\n{group_name} features:")
            print(mi_results.head(5))  # Show top 5 features in each group

print("\nEncoding complete. New dataframe shape:", df.shape)


# Box-Cox Transformations for numerical features

print("Implementing Box-Cox transformations for numerical features...")

# Dictionary to store lambda values
boxcox_store = {}

# Function to plot before and after transformation
def plot_transformation(df, original_col, transformed_col, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Original distribution
    sns.histplot(df[original_col].dropna(), kde=True, ax=ax1)
    ax1.set_title(f'Original {title}\nSkewness: {df[original_col].skew():.2f}')
    
    # Transformed distribution
    sns.histplot(df[transformed_col].dropna(), kde=True, ax=ax2)
    ax2.set_title(f'Box-Cox Transformed {title}\nSkewness: {df[transformed_col].skew():.2f}')
    
    plt.tight_layout()
    plt.show()


# 1. Transform Price (target variable)
print("\n1. Transforming Price (target variable)")
# Register Box-Cox transformer for Price
def price_transform(df):
    """
    Apply Box-Cox transformation to Price column in a DataFrame.
    
    Args:
        df: DataFrame containing a 'Price' column
    
    Returns:
        DataFrame with added 'Price_Transformed' column
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Extract the column to transform
    values = pd.to_numeric(df['Price'], errors='coerce').values
    
    # Handle NaN values
    mask = np.isnan(values)
    if mask.any():
        print(f"Warning: {mask.sum()} NaN values found in Price data. These will be imputed with median.")
        median_value = np.nanmedian(values)
        values[mask] = median_value
    
    # Add offset if needed to ensure all values are positive
    offset = 0
    if (values <= 0).any():
        offset = abs(values.min()) + 1
    values_offset = values + offset

    # Store offset in global dictionary
    boxcox_store["price_offset"] = offset

    # Initialize PowerTransformer
    pt = PowerTransformer(method="box-cox")

    # Apply transformation
    transformed = pt.fit_transform(values_offset.reshape(-1, 1)).flatten()

    # Store lambda value and the transformer in global dictionary
    boxcox_store["price_lambda"] = pt.lambdas_[0]
    boxcox_store["price_transformer"] = pt  # Store the transformer object directly

    # Add transformed column to the DataFrame
    df_copy["Price_Transformed"] = transformed
    
    return df_copy


price_transformer = register_transformer('price_boxcox', price_transform)
df = price_transformer.transform(df)
log('price_lambda', boxcox_store)
plot_transformation(df, 'Price', 'Price_Transformed', 'Price')


print("\n2. Transforming Landsize (high skewness)")
def landsize_transform(df):
    """
    Apply Box-Cox transformation to Landsize column in a DataFrame.
    
    Args:
        df: DataFrame containing a 'Landsize' column
    
    Returns:
        DataFrame with added 'Landsize_Transformed' column
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Extract the column to transform
    values = pd.to_numeric(df['Landsize'], errors='coerce').values
    
    # Handle NaN values
    mask = np.isnan(values)
    if mask.any():
        print(f"Warning: {mask.sum()} NaN values found in Landsize data. These will be imputed with median.")
        median_value = np.nanmedian(values)
        values[mask] = median_value
    
    # Add offset if needed to ensure all values are positive
    offset = 0
    if (values <= 0).any():
        offset = abs(values.min()) + 1
    values_offset = values + offset

    # Store offset in global dictionary
    boxcox_store["landsize_offset"] = offset

    # Initialize PowerTransformer
    pt = PowerTransformer(method="box-cox")

    # Apply transformation
    transformed = pt.fit_transform(values_offset.reshape(-1, 1)).flatten()

    # Store lambda value and the transformer in global dictionary
    boxcox_store["landsize_lambda"] = pt.lambdas_[0]
    boxcox_store["landsize_transformer"] = pt  # Store the transformer object directly
    
    # Add transformed column to the DataFrame
    df_copy['Landsize_Transformed'] = transformed
    
    return df_copy

landsize_transformer = register_transformer('landsize_boxcox', landsize_transform)
df = landsize_transformer.transform(df)
log('landsize_lambda', boxcox_store)
plot_transformation(df, 'Landsize', 'Landsize_Transformed', 'Landsize')


print("\n3. Transforming BuildingArea (high skewness)")
def building_area_transform(df):
    """
    Apply Box-Cox transformation to BuildingArea column in a DataFrame.
    
    Args:
        df: DataFrame containing a 'BuildingArea' column
    
    Returns:
        DataFrame with added 'BuildingArea_Transformed' column
    """

    df_copy = df.copy()
    
    # Extract the column to transform
    values = pd.to_numeric(df['BuildingArea'], errors='coerce').values
    
    # Handle NaN values
    mask = np.isnan(values)
    if mask.any():
        print(f"Warning: {mask.sum()} NaN values found in BuildingArea data. These will be imputed with median.")
        median_value = np.nanmedian(values)
        values[mask] = median_value
    
    # Add offset if needed to ensure all values are positive
    offset = 0
    if (values <= 0).any():
        offset = abs(values.min()) + 0.1
    values_offset = values + offset

    # Store offset in global dictionary
    boxcox_store["building_area_offset"] = offset

    # Initialize PowerTransformer
    pt = PowerTransformer(method="box-cox")

    # Apply transformation
    transformed = pt.fit_transform(values_offset.reshape(-1, 1)).flatten()

    # Store lambda value and the transformer in global dictionary
    boxcox_store["building_area_lambda"] = pt.lambdas_[0]
    boxcox_store["building_area_transformer"] = pt  # Store the transformer object directly
    
    # Add transformed column to the DataFrame
    df_copy['BuildingArea_Transformed'] = transformed
    
    return df_copy

    
building_area_transformer = register_transformer('building_area_boxcox', building_area_transform)
df = building_area_transformer.transform(df)
log('building_area_lambda', boxcox_store)
plot_transformation(df, 'BuildingArea', 'BuildingArea_Transformed', 'BuildingArea')


print("\n4. Transforming Distance")
def distance_transform(df):
    """
    Apply Box-Cox transformation to Distance column in a DataFrame.
    
    Args:
        df: DataFrame containing a 'Distance' column
    
    Returns:
        DataFrame with added 'Distance_Transformed' column
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Extract the column to transform
    values = pd.to_numeric(df['Distance'], errors='coerce').values
    
    # Handle NaN values
    mask = np.isnan(values)
    if mask.any():
        print(f"Warning: {mask.sum()} NaN values found in Distance data. These will be imputed with median.")
        median_value = np.nanmedian(values)
        values[mask] = median_value
    
    # Add offset if needed to ensure all values are positive
    offset = 0
    if (values <= 0).any():
        offset = abs(values.min()) + 0.1
    values_offset = values + offset

    # Store offset in global dictionary
    boxcox_store["distance_offset"] = offset

    # Initialize PowerTransformer
    pt = PowerTransformer(method="box-cox")

    # Apply transformation
    transformed = pt.fit_transform(values_offset.reshape(-1, 1)).flatten()

    # Store lambda value and the transformer in global dictionary
    boxcox_store["distance_lambda"] = pt.lambdas_[0]
    boxcox_store["distance_transformer"] = pt  # Store the transformer object directly
    
    # Add transformed column to the DataFrame
    df_copy['Distance_Transformed'] = transformed
    
    return df_copy
    
distance_transformer = register_transformer('distance_boxcox', distance_transform)
df = distance_transformer.transform(df)
log('distance_lambda', boxcox_store)
plot_transformation(df, 'Distance', 'Distance_Transformed', 'Distance')


def rooms_transform(df):
    """
    Apply Box-Cox transformation to Rooms column in a DataFrame.
    
    Args:
        df: DataFrame containing a 'Rooms' column
    
    Returns:
        DataFrame with added 'Rooms_Transformed' column
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Extract the column to transform
    values = pd.to_numeric(df['Rooms'], errors='coerce').values
    
    # Handle NaN values
    mask = np.isnan(values)
    if mask.any():
        print(f"Warning: {mask.sum()} NaN values found in Rooms data. These will be imputed with median.")
        median_value = np.nanmedian(values)
        values[mask] = median_value

    
    # Add offset if needed to ensure all values are positive
    offset = 0
    if (values <= 0).any():
        offset = abs(values.min()) + 0.1
    values_offset = values + offset
    # Store offset in global dictionary
    boxcox_store["rooms_offset"] = offset

    # Initialize PowerTransformer
    pt = PowerTransformer(method="box-cox")

    # Apply transformation
    transformed = pt.fit_transform(values_offset.reshape(-1, 1)).flatten()

    # Store lambda value and the transformer in global dictionary
    boxcox_store["rooms_lambda"] = pt.lambdas_[0]
    boxcox_store["rooms_transformer"] = pt  # Store the transformer object directly
    
    # Add transformed column to the DataFrame
    df_copy['Rooms_Transformed'] = transformed
    
    return df_copy



rooms_transformer = register_transformer('rooms_boxcox', rooms_transform)
df = rooms_transformer.transform(df)
log('rooms_lambda', boxcox_store)
plot_transformation(df, 'Rooms', 'Rooms_Transformed', 'Rooms')


def bathroom_transform(df):
    """
    Apply Box-Cox transformation to Bathroom column in a DataFrame.
    
    Args:
        df: DataFrame containing a 'Bathroom' column
    
    Returns:
        DataFrame with added 'Bathroom_Transformed' column
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Extract the column to transform
    values = pd.to_numeric(df['Bathroom'], errors='coerce').values
    
    # Handle NaN values
    mask = np.isnan(values)
    if mask.any():
        print(f"Warning: {mask.sum()} NaN values found in Bathroom data. These will be imputed with median.")
        median_value = np.nanmedian(values)
        values[mask] = median_value
    
    # Initialize PowerTransformer
    pt = PowerTransformer(method='box-cox')
    
    # Add offset if needed to ensure all values are positive
    offset = 0
    if (values <= 0).any():
        offset = abs(values.min()) + 1
    values_offset = values + offset

    # Store offset in global dictionary
    boxcox_store["bathroom_offset"] = offset

    # Initialize PowerTransformer
    pt = PowerTransformer(method="box-cox")

    # Apply transformation
    transformed = pt.fit_transform(values_offset.reshape(-1, 1)).flatten()

    # Store lambda value and the transformer in global dictionary
    boxcox_store["bathroom_lambda"] = pt.lambdas_[0]
    boxcox_store["bathroom_transformer"] = pt  # Store the transformer object directly
    
    # Add transformed column to the DataFrame
    df_copy['Bathroom_Transformed'] = transformed
    
    return df_copy


bathroom_transformer = register_transformer('bathroom_boxcox', bathroom_transform)
df = bathroom_transformer.transform(df)
log('bathroom_lambda', boxcox_store)
plot_transformation(df, 'Bathroom', 'Bathroom_Transformed', 'Bathroom')


def car_transform(df):
    """
    Apply Box-Cox transformation to Car column in a DataFrame.
    
    Args:
        df: DataFrame containing a 'Car' column
    
    Returns:
        DataFrame with added 'Car_Transformed' column
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Extract the column to transform
    values = pd.to_numeric(df['Car'], errors='coerce').values
    
    # Handle NaN values
    mask = np.isnan(values)
    if mask.any():
        print(f"Warning: {mask.sum()} NaN values found in Car data. These will be imputed with median.")
        median_value = np.nanmedian(values)
        values[mask] = median_value
    
    
    # Add offset if needed to ensure all values are positive
    offset = 0
    if (values <= 0).any():
        offset = abs(values.min()) + 1
    values_offset = values + offset

    # Store offset in global dictionary
    boxcox_store["car_offset"] = offset

    # Initialize PowerTransformer
    pt = PowerTransformer(method="box-cox")

    # Apply transformation
    transformed = pt.fit_transform(values_offset.reshape(-1, 1)).flatten()

    # Store lambda value and the transformer in global dictionary
    boxcox_store["car_lambda"] = pt.lambdas_[0]
    boxcox_store["car_transformer"] = pt  # Store the transformer object directly
    
    # Add transformed column to the DataFrame
    df_copy['Car_Transformed'] = transformed
    
    return df_copy

car_transformer = register_transformer('car_boxcox', car_transform)
df = car_transformer.transform(df)
log('car_lambda', boxcox_store)
plot_transformation(df, 'Car', 'Car_Transformed', 'Car')



def property_age_transform(df):
    """
    Apply Box-Cox transformation to PropertyAge column in a DataFrame.
    
    Args:
        df: DataFrame containing a 'PropertyAge' column
    
    Returns:
        DataFrame with added 'PropertyAge_Transformed' column
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Extract the column to transform
    values = pd.to_numeric(df['PropertyAge'], errors='coerce').values
    
    # Handle NaN values
    mask = np.isnan(values)
    if mask.any():
        print(f"Warning: {mask.sum()} NaN values found in PropertyAge data. These will be imputed with median.")
        median_value = np.nanmedian(values)
        values[mask] = median_value
    
    # Add offset if needed to ensure all values are positive
    offset = 0
    if (values <= 0).any():
        offset = abs(values.min()) + 1
    values_offset = values + offset
    # Store offset in global dictionary
    boxcox_store["propertyage_offset"] = offset

    # Initialize PowerTransformer
    pt = PowerTransformer(method="box-cox")

    # Apply transformation
    transformed = pt.fit_transform(values_offset.reshape(-1, 1)).flatten()

    # Store lambda value and the transformer in global dictionary
    boxcox_store["propertyage_lambda"] = pt.lambdas_[0]
    boxcox_store["propertyage_transformer"] = pt  # Store the transformer object directly
    
    # Add transformed column to the DataFrame
    df_copy['PropertyAge_Transformed'] = transformed
    
    return df_copy

print("\n6. Transforming other numerical variables")
if 'PropertyAge' in df.columns and abs(df['PropertyAge'].skew()) > 0.5:
    property_age_transformer = register_transformer('propertyage_boxcox', property_age_transform)
    df = property_age_transformer.transform(df)
    log('propertyage_lambda', boxcox_store)
    plot_transformation(df, 'PropertyAge', 'PropertyAge_Transformed', 'PropertyAge')



# Decide which columns to keep (original vs transformed)
print("\nDeciding which columns to keep (original vs transformed)...")

# Calculate skewness before and after transformation
skewness_comparison = []
for col in df.columns:
    if col.endswith('_Transformed') and col.replace('_Transformed', '') in df.columns:
        original_col = col.replace('_Transformed', '')
        original_skew = df[original_col].skew()
        transformed_skew = df[col].skew()
        improvement = abs(original_skew) - abs(transformed_skew)
        skewness_comparison.append({
            'Original Column': original_col,
            'Original Skewness': original_skew,
            'Transformed Column': col,
            'Transformed Skewness': transformed_skew,
            'Improvement': improvement
        })
        
# Display skewness comparison
skewness_df = pd.DataFrame(skewness_comparison).sort_values('Improvement', ascending=False)
print("\nSkewness comparison (before vs after transformation):")
print(skewness_df)



def apply_transformers_with_error_handling(df, transformers):
    """
    Apply each transformer sequentially to the DataFrame and catch any errors.
    
    Args:
        df: Input DataFrame
        transformers: Dictionary of transformer name to transformer object
    
    Returns:
        Transformed DataFrame or None if error occurred
    """
    df_transformed = df.copy()
    
    for name, transformer in transformers.items():
        try:
            print(f"Applying transformer: {name}")
            df_transformed = transformer.fit_transform(df_transformed)
            print(f"✓ Successfully applied {name}")
        except Exception as e:
            print(f"\n❌ ERROR in transformer: {name}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nFull stack trace:")
            traceback.print_exc(file=sys.stdout)
            print(f"\nDataFrame state before failing transformer:")
            print(f"Shape: {df_transformed.shape}")
            display(df_transformed.head())
            return None
    
    return df_transformed

# Load your data
df_new = pd.read_csv("Datasets/01_Melbourne_Residential.csv")

# Apply transformers with error handling
result_df = apply_transformers_with_error_handling(df_new, transformers)

if result_df is not None:
    print("\nAll transformers applied successfully!")
    print(f"Final DataFrame shape: {result_df.shape}")
    display(result_df.head())
    display(df.head())



# Convert your transformer dictionary to a Pipeline
def create_pipeline_from_transformers(transformers_dict):
    """
    Convert a dictionary of transformers to a scikit-learn Pipeline
    
    Args:
        transformers_dict: Dictionary of transformer name to transformer object
    
    Returns:
        sklearn.pipeline.Pipeline object
    """
    steps = [(name, transformer) for name, transformer in transformers_dict.items()]
    return Pipeline(steps)
# Create a pipeline
pipeline = create_pipeline_from_transformers(transformers)

# Load data
df_new_new = pd.read_csv("Datasets/01_Melbourne_Residential.csv")

# Apply transformations
result_df_new = pipeline.fit_transform(df_new_new)

if result_df is not None:
    print("\nAll transformations applied successfully!")
    print(f"Final DataFrame shape: {result_df_new.shape}")
    display(result_df_new.head())
    
    # Save the pipeline
    with open("melbourne_pipeline.pkl", "wb") as f:
        try:
            pickle.dump(pipeline, f)
            print("Pipeline successfully saved!")
        except Exception as e:
            print(f"Error saving pipeline: {str(e)}")
            
            # Try to identify which transformer is causing the issue
            for name, transformer in transformers.items():
                try:
                    with open(f"test_{name}.pkl", "wb") as test_f:
                        pickle.dump(transformer, test_f)
                except Exception as e:
                    print(f"Transformer '{name}' is not picklable: {str(e)}")



# First, let's convert the Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')  # Assuming day/month/year format

# Sort the dataframe by date
df_sorted = df.sort_values('Date')

# Let's see the date range in our dataset
print(f"Date range: {df_sorted['Date'].min()} to {df_sorted['Date'].max()}")


# Create price bins for stratification (adjust the number of bins as needed)
df['price_bin'] = pd.qcut(df['Price'], q=5, labels=False)

# Create a combined stratification variable
if 'PropertyType' in df.columns:
    df['strat_var'] = df['PropertyType'].astype(str) + '_' + df['price_bin'].astype(str)
else:
    df['strat_var'] = df['price_bin']

# Create time periods (e.g., months or quarters)
df['time_period'] = pd.qcut(df['Date'].astype(int), q=10, labels=False)

# Define  feature columns and target
feature_cols = [col for col in df.columns if col not in ['Price', 'Price_Transformed', 'time_period', 'price_bin', 'strat_var', 'Address', 'Date', 'Postcode', 'Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'PropertyAge', 'DistanceToCBD']]
target = 'Price_Transformed' if 'Price_Transformed' in df.columns else 'Price'

# Initialize empty lists for train and validation indices
train_indices = []
val_indices = []

# For each time period, perform stratified sampling
for period in df['time_period'].unique():
    period_df = df[df['time_period'] == period]
    period_indices = period_df.index
    
    # If the period has enough samples for stratification
    if len(period_df['strat_var'].unique()) > 1 and len(period_df) >= 10:
        train_idx, val_idx = train_test_split(
            period_indices,
            test_size=0.2,
            random_state=42,
            stratify=period_df['strat_var']
        )
    else:
        # For small periods or when stratification isn't possible, just do a regular split
        train_idx, val_idx = train_test_split(
            period_indices,
            test_size=0.2,
            random_state=42
        )
    
    train_indices.extend(train_idx)
    val_indices.extend(val_idx)
log("test_size", 0.2)
log("random_state", 42)
log("stratification_method", "price_bin_and_property_type")
log("time_periods", 10)
# Create the final datasets
train_df = df.loc[train_indices].copy()
val_df = df.loc[val_indices].copy()



# Create the final datasets
X_train = train_df[feature_cols]
X_val = val_df[feature_cols]
y_train = train_df[target]
y_val = val_df[target]

log("training_set_size", len(train_df))
log("validation_set_size", len(val_df))
log("training_date_min", train_df['Date'].min().strftime('%Y-%m-%d'))
log("training_date_max", train_df['Date'].max().strftime('%Y-%m-%d'))
log("validation_date_min", val_df['Date'].min().strftime('%Y-%m-%d'))
log("validation_date_max", val_df['Date'].max().strftime('%Y-%m-%d'))


# Check stratification
print("\nTraining set price bin distribution:")
print(df.loc[train_indices, 'price_bin'].value_counts(normalize=True))
print("\nValidation set price bin distribution:")
print(df.loc[val_indices, 'price_bin'].value_counts(normalize=True))

if 'PropertyType' in df.columns:
    print("\nTraining set property type distribution:")
    print(df.loc[train_indices, 'PropertyType'].value_counts(normalize=True))
    print("\nValidation set property type distribution:")
    print(df.loc[val_indices, 'PropertyType'].value_counts(normalize=True))

# Optional: Check if the time ordering is preserved
train_dates = df.loc[train_indices, 'Date']
val_dates = df.loc[val_indices, 'Date']

val_dates_after_train = (val_dates > train_dates.max()).mean() * 100
train_dates_before_val = (train_dates < val_dates.min()).mean() * 100
log("pct_val_dates_after_max_train", f"{val_dates_after_train:.2f}%")
log("pct_train_dates_before_min_val", f"{train_dates_before_val:.2f}%")


# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on validation set
val_predictions = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, val_predictions)

print(mlflow.log_metric("validation_rmse", rmse))
print(mlflow.log_metric("validation_r2", r2))
print(mlflow.log_metric("validation_mse", mse))


# Visualize predictions vs actual prices
plt.figure(figsize=(10, 6))
plt.scatter(y_val, val_predictions, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices on Validation Set')
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
mlflow.log_artifact("actual_vs_predicted.png")


# Plot residuals
residuals = y_val - val_predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_val, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Actual Price')
plt.ylabel('Residual')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig("residuals.png")
mlflow.log_artifact("residuals.png")


# Plot predictions over time
plt.figure(figsize=(15, 8))
plt.scatter(val_df['Date'], y_val, alpha=0.5, s=20, label='Actual', color='blue')
plt.scatter(val_df['Date'], val_predictions, alpha=0.5, s=20, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Price_Transformed')
plt.title('Price Trends Over Time (Validation Set)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("time_series.png")
mlflow.log_artifact("time_series.png")


# Feature importance
coefficients = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_
})
coefficients = coefficients.sort_values('Coefficient', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(coefficients['Feature'][:15], coefficients['Coefficient'][:15])
plt.xlabel('Coefficient')
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.savefig("feature_importance.png")
mlflow.log_artifact("feature_importance.png")



# Prepare data for PyCaret
# Combine features and target into a single dataframe for training
train_data = train_df[feature_cols + [target]].copy()
val_data = val_df[feature_cols + [target]].copy()

# Log dataset information
log("pycaret_train_samples", len(train_data))
log("pycaret_val_samples", len(val_data))
log("pycaret_features", len(feature_cols))
log("pycaret_train_cols", list(train_data))


# Initialize PyCaret setup with MLflow tracking
# Note: PyCaret automatically logs to MLflow when log_experiment=True
reg_setup = setup(
    data=train_data,
    target=target,
    session_id=42,
    log_experiment=True,
    experiment_name="Housing Price Prediction",
    log_plots=True,
    verbose=True,
    ignore_features=['time_period', 'price_bin', 'strat_var'] if any(col in train_data.columns for col in ['time_period', 'price_bin', 'strat_var']) else None,
    fold_strategy='timeseries',  # Use time series cross-validation
    data_split_shuffle=False,
    fold_shuffle=False,
    fold=5
)

# Log setup parameters
log("pycaret_normalize", True)
log("pycaret_transformation", True)
log("pycaret_fold_strategy", "timeseries")
log("pycaret_folds", 5)


# Compare models and get the best models table
best_models = compare_models(
    n_select=5,  # Select top 5 models
    sort='RMSE',  # Sort by RMSE
    verbose=True
)

# If best_models is a single model (not a list)
if not isinstance(best_models, list):
    best_models = [best_models]


# Tune each of the top models
tuned_models = []
for i, model in enumerate(best_models):   
    model_name = model.__class__.__name__ 
    print(f"\nTuning {model_name}...")    
                                          
    # Log the base model name             
    log(f"top_model_{i+1}", model_name)   
    
    # Tune the model
    tuned_model = tune_model(
        model,
        optimize='RMSE',
        n_iter=10,
        search_library='optuna',
    )
    
    tuned_models.append(tuned_model)
    
    # Evaluate on validation set
    pred_holdout = predict_model(tuned_model, data=val_data)
    val_rmse = np.sqrt(mean_squared_error(pred_holdout[target], pred_holdout['prediction_label']))
    val_r2 = r2_score(pred_holdout[target], pred_holdout['prediction_label'])
    
    # Log validation metrics
    mlflow.log_metric(f"val_rmse_{model_name}", val_rmse)
    mlflow.log_metric(f"val_r2_{model_name}", val_r2)
    
    # Create and log residual plot
    plt.figure(figsize=(10, 6))
    residuals = pred_holdout[target] - pred_holdout['prediction_label']
    plt.scatter(pred_holdout[target], residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Price')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {model_name}')
    plt.tight_layout()
    plt.savefig(f"residuals_{model_name}.png")
    mlflow.log_artifact(f"residuals_{model_name}.png")
    
    # Create and log actual vs predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(pred_holdout[target], pred_holdout['prediction_label'], alpha=0.5)
    plt.plot([pred_holdout[target].min(), pred_holdout[target].max()], 
             [pred_holdout[target].min(), pred_holdout[target].max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.tight_layout()
    plt.savefig(f"actual_vs_predicted_{model_name}.png")
    mlflow.log_artifact(f"actual_vs_predicted_{model_name}.png")


# Select the best model based on validation RMSE
best_idx = np.argmin([
    np.sqrt(mean_squared_error(
        predict_model(model, data=val_data)[target], 
        predict_model(model, data=val_data)['prediction_label']
    )) 
    for model in tuned_models
])

best_model = tuned_models[best_idx]
best_model_name = best_model.__class__.__name__ 


# Log the final best model
log("best_model", best_model_name)
final_model = finalize_model(best_model)


# Save the model with PyCaret (this also logs to MLflow)
save_model(final_model, 'housing_price_prediction')

model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="housing_price_prediction"
)


def predict(input_data):
    """
    Predict the original property price based on input data.

    Args:
        input_data: Dictionary or tuple of dictionaries containing property features
        
    Returns:
        Predicted property price(s) in original scale ($)
    """
    # Convert input data to DataFrame
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, tuple):
        input_df = pd.DataFrame(list(input_data))
    else:
        input_df = pd.DataFrame(input_data)
    
    # Apply necessary preprocessing to match the model's expected features
    processed_df = preprocess_for_prediction(input_df)
    
    # Generate predictions using the PyCaret model
    prediction_df = predict_model(final_model, data=processed_df)
    
    # Extract the transformed predictions
    predicted_transformed_values = prediction_df["prediction_label"].values
    print("predicted_transformed", predicted_transformed_values)
    
    # Inverse-transform to get original price
    original_prices = [inverse_transform_price(val) for val in predicted_transformed_values]
    
    # Return single value or list based on input
    if len(original_prices) == 1:
        return original_prices[0]
    else:
        return original_prices


def preprocess_for_prediction(df):
    """
    Apply all necessary preprocessing steps to prepare data for prediction.

    Args:
        df: DataFrame with raw input features
        
    Returns:
        DataFrame with all features required by the model
    """
    processed = df.copy()
    
    # 1. Property Type encoding
    if "Type" in processed.columns:
        # Map text property types to codes if needed
        type_mapping = {"House": "h", "Unit/Apartment": "u", "Townhouse": "t"}
        if not processed["Type"].isin(["h", "u", "t"]).all():
            processed["Type"] = processed["Type"].map(lambda x: type_mapping.get(x, x))
        
        # Create one-hot encoded features
        processed["PropType_House"] = (processed["Type"] == "h").astype(int)
        processed["PropType_Townhouse"] = (processed["Type"] == "t").astype(int)
        processed["PropType_Unit/Apartment"] = (processed["Type"] == "u").astype(int)
        processed = processed.drop("Type", axis=1)
    
    # 2. Method encoding
    if "Method" in processed.columns:
        # Create one-hot encoded Method features
        method_columns = ["Method_PI", "Method_S", "Method_SA", "Method_SP", "Method_VB"]
        for col in method_columns:
            method_code = col.split("_")[1]
            processed[col] = (processed["Method"] == method_code).astype(int)
        processed = processed.drop("Method", axis=1)
    
    # 3. Suburb encoding
    if "Suburb" in processed.columns:
        # Use the suburb_to_rank_dict from the transformer
        suburb_ranks = transform_suburb.encoder.suburb_to_rank_dict
        processed["Suburb_PriceRank"] = processed["Suburb"].map(
            suburb_ranks, na_action='ignore')
        # Fill missing values with median rank
        if processed["Suburb_PriceRank"].isna().any():
            median_rank = np.median(list(suburb_ranks.values()))
            processed["Suburb_PriceRank"] = processed["Suburb_PriceRank"].fillna(median_rank)
        processed = processed.drop("Suburb", axis=1)
    
    # 4. Seller encoding
    if "Seller" in processed.columns:
        # Create one-hot encoded Seller features
        seller_cols = [c for c in df.columns if c.startswith("Seller_")]
        common_sellers = ["Barry", "Biggin", "Brad", "Buxton", "Fletchers", "Gary", 
                         "Greg", "Harcourts", "Hodges", "Jas", "Jellis", "Kay", 
                         "Love", "Marshall", "McGrath", "Miles", "Nelson", "Noel", 
                         "RT", "Raine", "Ray", "Stockdale", "Sweeney", "Village", 
                         "Williams", "Woodards", "YPA", "hockingstuart"]
        
        # Initialize all seller columns to 0
        for seller in common_sellers:
            processed[f"Seller_{seller}"] = 0
        
        # Set the appropriate column to 1 or "Other" if not in common sellers
        for idx, seller in enumerate(processed["Seller"]):
            if seller in common_sellers:
                processed.loc[idx, f"Seller_{seller}"] = 1
            else:
                processed.loc[idx, "Seller_Other"] = 1
                
        processed = processed.drop("Seller", axis=1)
    
    # 5. Direction features if needed
    if "Direction" in processed.columns:
        direction_cols = ["Direction_N", "Direction_S", "Direction_E", "Direction_W"]
        for dir_col in direction_cols:
            dir_code = dir_col.split("_")[1]
            processed[dir_col] = (processed["Direction"] == dir_code).astype(int)
        processed = processed.drop("Direction", axis=1)
    
    # 6. Numerical transformations
    # Apply Box-Cox transformations to numeric features
    if "Landsize" in processed.columns:
        processed["Landsize_Transformed"] = box_cox_transform(
            processed["Landsize"], 
            boxcox_store["landsize_lambda"], 
            boxcox_store.get("landsize_offset", 0)
        )
    
    if "BuildingArea" in processed.columns:
        processed["BuildingArea_Transformed"] = box_cox_transform(
            processed["BuildingArea"], 
            boxcox_store["building_area_lambda"], 
            boxcox_store.get("building_area_offset", 0)
        )
    
    if "Distance" in processed.columns:
        processed["Distance_Transformed"] = box_cox_transform(
            processed["Distance"], 
            boxcox_store["distance_lambda"], 
            boxcox_store.get("distance_offset", 0)
        )
    
    if "Rooms" in processed.columns:
        processed["Rooms_Transformed"] = box_cox_transform(
            processed["Rooms"], 
            boxcox_store["rooms_lambda"], 
            boxcox_store.get("rooms_offset", 0)
        )
    
    if "Bathroom" in processed.columns:
        processed["Bathroom_Transformed"] = box_cox_transform(
            processed["Bathroom"], 
            boxcox_store["bathroom_lambda"], 
            boxcox_store.get("bathroom_offset", 0)
        )
    
    if "Car" in processed.columns:
        processed["Car_Transformed"] = box_cox_transform(
            processed["Car"], 
            boxcox_store["car_lambda"], 
            boxcox_store.get("car_offset", 0)
        )
    
    if "PropertyAge" in processed.columns:
        processed["PropertyAge_Transformed"] = box_cox_transform(
            processed["PropertyAge"], 
            boxcox_store["propertyage_lambda"], 
            boxcox_store.get("propertyage_offset", 0)
        )
    
    # Convert boolean values to integers
    bool_columns = processed.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        processed[col] = processed[col].astype(int)
    
    # 7. Add any missing columns required by the model with default values (0)
    required_columns = ['PropType_House', 'PropType_Townhouse', 'PropType_Unit/Apartment', 
                        'Method_PI', 'Method_S', 'Method_SA', 'Method_SP', 'Method_VB', 
                        'Suburb_PriceRank', 'Seller_Barry', 'Seller_Biggin', 'Seller_Brad', 
                        'Seller_Buxton', 'Seller_Fletchers', 'Seller_Gary', 'Seller_Greg', 
                        'Seller_Harcourts', 'Seller_Hodges', 'Seller_Jas', 'Seller_Jellis', 
                        'Seller_Kay', 'Seller_Love', 'Seller_Marshall', 'Seller_McGrath', 
                        'Seller_Miles', 'Seller_Nelson', 'Seller_Noel', 'Seller_Other', 
                        'Seller_RT', 'Seller_Raine', 'Seller_Ray', 'Seller_Stockdale', 
                        'Seller_Sweeney', 'Seller_Village', 'Seller_Williams', 'Seller_Woodards', 
                        'Seller_YPA', 'Seller_hockingstuart', 'Landsize_Transformed', 
                        'BuildingArea_Transformed', 'Distance_Transformed', 'Rooms_Transformed', 
                        'Bathroom_Transformed', 'Car_Transformed', 'PropertyAge_Transformed']
    
    for col in required_columns:
        if col not in processed.columns:
            processed[col] = 0    
    return processed


def box_cox_transform(values, lambda_val, offset=0):
    """Apply Box-Cox transformation to a series of values"""
    values = pd.to_numeric(values, errors='coerce')
    # Handle NaNs
    values = values.fillna(values.median())

    # Apply offset if needed
    values_offset = values + offset
    
    # Apply transformation
    if abs(lambda_val) < 1e-10:  # lambda is close to zero
        return np.log(values_offset)
    else:
        return ((values_offset ** lambda_val) - 1) / lambda_val


def inverse_transform_price(transformed_value):
    """
    Inverse-transform a price value using the stored PowerTransformer.

    Args:
        transformed_value: Transformed price value
        
    Returns:
        Original price value
    """
    # Get the offset
    offset = boxcox_store.get("price_offset", 0)
    
    # Get the transformer from boxcox_store
    if "price_transformer" in boxcox_store:
        pt = boxcox_store["price_transformer"]
        
        # Reshape for scikit-learn
        value_reshaped = np.array([transformed_value]).reshape(-1, 1)
        
        # Use the transformer's built-in inverse_transform method
        original_with_offset = pt.inverse_transform(value_reshaped)[0][0]
        
        # Remove the offset
        original_price = original_with_offset - offset
        
        return original_price
    else:
        # Fallback to manual implementation if transformer isn't available
        print("Warning: PowerTransformer not found, using fallback method")
        lambda_val = boxcox_store["price_lambda"]
        
        if abs(lambda_val) < 1e-10:  # lambda is close to zero
            x_original = np.exp(transformed_value)
        else:
            x_original = np.power(lambda_val * transformed_value + 1, 1/lambda_val)
        
        return x_original - offset


input_data = (
    {
        "Suburb": "Reservoir",
        "Rooms": 3,
        "Type": "House",
        "Method": "S",
        "Seller": "Ray",
        "Distance": 11.2,
        "Bathroom": 1.0,
        "Car": 2,
        "Landsize": 556.0,
        "BuildingArea": 120.0,
        "PropertyAge": 50,
        "Direction": "N",
        "LandSizeNotOwned": False,
    },
)

predict(input_data)


# Feature importance for the best model (if available)
plt.figure(figsize=(12, 8))
importance_fig = plot_model(final_model, plot='feature', save=True)
importance_fig = plot_model(final_model, plot='feature')
mlflow.log_artifact('Feature Importance.png')


# Log the final model's pipeline steps
pipeline_steps = []
for step in final_model.steps:
    step_name = step[0]
    step_estimator = step[1].__class__.__name__
    pipeline_steps.append(f"{step_name}: {step_estimator}")

log("model_pipeline", " → ".join(pipeline_steps))


# Log hyperparameters of the final model
try:
    final_estimator = final_model.steps[-1][1]
    for param, value in final_estimator.get_params().items():
        if not param.startswith('_'):
            try:
                # Only log simple types that can be serialized
                if isinstance(value, (int, float, str, bool)) or value is None:
                    log(f"param_{param}", value)
            except:
                pass
except:
    print("Could not log all hyperparameters")


# After you've finalized your model (around line 150)
# Save the model with PyCaret (this also logs to MLflow)
save_model(final_model, "final_housing_price_model")

# Save the model to BentoML
# First, get the preprocessing pipeline and model separately
model_pipeline = final_model

# Save to BentoML with appropriate signatures
bento_model = bentoml.sklearn.save_model(
    "housing_price_model",
    model_pipeline,
    signatures={
        "predict": {
            "batchable": True,
            "batch_dim": 0,
        },
    },
    custom_objects={
        "feature_names": feature_cols,
        "target_name": target,
        "model_type": best_model_name,
        "boxcox_store": boxcox_store,
        "major_sellers": major_sellers,
        "suburb_to_rank_dict": transform_suburb.encoder.suburb_to_rank_dict,
    },
    metadata={
        "description": "Housing price prediction model trained with PyCaret",
        "model_framework": "PyCaret + " + best_model_name,
        "dataset_size": len(train_data) + len(val_data),
        "feature_count": len(feature_cols),
    }
)

print(f"Model saved to BentoML: {bento_model}")
print(f"Model tag: {bento_model.tag}")






!bentoml models get housing_price_model


!bentoml models push housing_price_model


!bentoml deploy service


!python service/test.py



