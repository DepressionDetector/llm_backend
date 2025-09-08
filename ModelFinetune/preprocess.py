import pandas as pd
import os
import json

# Load your dataset (works for CSV or Excel)
file_path = "Resources/PHQ9_Student_Depression_Dataset.xlsx"   # fixed path
df = pd.read_excel(file_path)

# Create new columns
df["instruction"] = "Analyze the following PHQ-9 responses and provide the score and depression level."

# Combine all PHQ answers into one text block (PHQ1..PHQ9)
df["input"] = df.apply(
    lambda row: "\n".join([f"{i}. {row[f'PHQ{i}_Text']}" for i in range(1, 10) if pd.notna(row.get(f'PHQ{i}_Text'))]),
    axis=1
)

# Output column (score + level)
df["output"] = df.apply(
    lambda row: f"PHQ-9 Score: {row['PHQ-9 Score']}\nDepression Level: {row['Depression Level']}",
    axis=1
)

# Select only fine-tune columns
final_df = df[["instruction", "input", "output"]]

# Ensure output directory exists
os.makedirs("Output_dir", exist_ok=True)

# Save as CSV
csv_path = "Output_dir/Preprocessed_dataset.csv"
final_df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"✅ Fine-tune CSV saved as {csv_path}")

# Save as JSONL
jsonl_path = "Output_dir/Preprocessed_dataset.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for _, row in final_df.iterrows():
        record = {
            "instruction": row["instruction"],
            "input": row["input"],
            "output": row["output"]
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ Fine-tune JSONL saved as {jsonl_path}")

# Preview first 5 rows
print(final_df.head())
