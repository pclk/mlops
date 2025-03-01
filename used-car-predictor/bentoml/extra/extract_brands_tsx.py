import pandas as pd

# Load CSV file
df = pd.read_csv("../brand_model_frequency.csv")

# Filter brand models with frequency >= 2
filtered_df = df[df["Frequency"] >= 2]

# Extract and sort brand model names alphabetically
brand_models = sorted(filtered_df["Brand_Model"].tolist())

# Format as a TypeScript array
ts_array = "const brandModels: string[] = [\n" + ",\n".join([f'    "{bm}"' for bm in brand_models]) + "\n];\n\nexport default brandModels;"

# Save to file
output_path = "brand_models.ts"
with open(output_path, "w") as f:
    f.write(ts_array)

print(f"✅ TypeScript array saved to {output_path}:\n")
print(ts_array)
