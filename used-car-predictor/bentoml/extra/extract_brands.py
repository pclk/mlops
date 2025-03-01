import pandas as pd

df = pd.read_csv("../brand_model_frequency.csv")

# Filter brands with frequency >= 2
filtered_df = df[df["Frequency"] >= 2]

# Convert to dictionary (Brand_Model -> Frequency)
brand_dict = dict(zip(filtered_df["Brand_Model"], filtered_df["Frequency"]))

# Format as a string for easy copy-paste
formatted_dict = "{\n" + ",\n".join([f'    "{k}": {v}' for k, v in brand_dict.items()]) + "\n}"

output_path = "filtered_brand_dict.txt"
with open(output_path, "w") as f:
    f.write(formatted_dict)

print(f"Dictionary saved to {output_path}:\n")
print(formatted_dict)
