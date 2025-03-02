#!/usr/bin/env python3

import pandas as pd
import os

# Create output directory if it doesn't exist
output_dir = "./ts-data"
os.makedirs(output_dir, exist_ok=True)

dataset_path = "../../datasets/01_Melbourne_Residential.csv"
df = pd.read_csv(dataset_path)

# Print columns to verify names
print("Dataset columns:", df.columns.tolist())

# Define mappings between dataset columns and TypeScript
# Adjust column names to match your actual dataset
mappings = [
    {"column": "Suburb", "ts_var": "suburbs", "output_file": "suburbs.ts"},
    {"column": "Type", "ts_var": "propertyTypes", "output_file": "propertyTypes.ts"},
    {
        "column": "Method",
        "ts_var": "sellingMethods",
        "output_file": "sellingMethods.ts",
    },
    {"column": "Seller", "ts_var": "sellers", "output_file": "sellers.ts"},
    # If direction exists in your dataset, uncomment and adjust column name:
    # {'column': 'Direction', 'ts_var': 'directions', 'output_file': 'directions.ts'},
]

for mapping in mappings:
    column = mapping["column"]

    if column in df.columns:
        # Extract unique values and sort them
        unique_values = sorted(df[column].dropna().unique())

        # Format as a TypeScript array
        ts_var = mapping["ts_var"]
        ts_array = (
            f"const {ts_var}: string[] = [\n"
            + ",\n".join([f'    "{value}"' for value in unique_values])
            + f"\n];\n\nexport default {ts_var};"
        )

        # Save to file
        output_path = os.path.join(output_dir, mapping["output_file"])
        with open(output_path, "w") as f:
            f.write(ts_array)

        print(f"✅ TypeScript array saved to {output_path}")
    else:
        print(f"⚠️ Column '{column}' not found in the dataset")
