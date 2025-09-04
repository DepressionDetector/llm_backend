import pandas as pd

# Load your dataset (works for CSV or Excel)
file_path = "Resources\PHQ9_Student_Depression_Dataset.xlsx"   # change if Excel
df = pd.read_excel(file_path)
 # or pd.read_excel(file_path)

# Create new columns
df["instruction"] = "Analyze the following PHQ-9 responses and provide the score and depression level."

# Combine all PHQ answers into one text block
df["input"] = df.apply(
    lambda row: "\n".join([f"{i+1}. {row[f'PHQ{i+1}_Text']}" for i in range(9)]),
    axis=1
)

# Output column (score + level)
df["output"] = df.apply(
    lambda row: f"PHQ-9 Score: {row['PHQ-9 Score']}\nDepression Level: {row['Depression Level']}",
    axis=1
)

# Select only fine-tune columns
final_df = df[["instruction", "input", "output"]]

# Save as new CSV
output_path = "Output_dir/Preprocessed dataset.csv"
final_df.to_csv(output_path, index=False, encoding="utf-8")

print(f"✅ Fine-tune CSV saved as {output_path}")

print(final_df.head())

