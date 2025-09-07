import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("Output_dir\Preprocessed dataset.csv")  # or read_csv if CSV
print("📂 Columns in dataset:", df.columns.tolist())
print("🔝 First few rows:")
print(df.head())

# -------------------------------
# Load paraphrasing model
# -------------------------------
paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

def paraphrase_text(text, num_return_sequences=2, num_beams=4):
    """Generate paraphrased versions of a given text"""
    if not isinstance(text, str) or text.strip() == "":
        return [text]
    try:
        outputs = paraphraser(
            f"paraphrase: {text}",
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            max_length=256,
            clean_up_tokenization_spaces=True
        )
        return [o["generated_text"] for o in outputs]
    except Exception as e:
        print(f"⚠️ Error paraphrasing: {e}")
        return [text]

# -------------------------------
# Augment dataset
# -------------------------------
augmented_rows = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Paraphrasing rows"):
    input_paraphrases = paraphrase_text(row["input"])
    instr_paraphrases = paraphrase_text(row["instruction"])

    if i < 3:  # show a few examples for debugging
        print(f"\nRow {i}:")
        print(" Original input:", row["input"])
        print(" Paraphrased inputs:", input_paraphrases)
        print(" Original instruction:", row["instruction"])
        print(" Paraphrased instructions:", instr_paraphrases)

    for inp in input_paraphrases:
        for instr in instr_paraphrases:
            augmented_rows.append({
                "instruction": instr,
                "input": inp,
                "output": row["output"]
            })

# -------------------------------
# Combine original + augmented
# -------------------------------
df_augmented = pd.DataFrame(augmented_rows)
df_final = pd.concat([df, df_augmented], ignore_index=True)

# -------------------------------
# Save CSV
# -------------------------------
output_file = "Augmented_dataset.csv"
df_final.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n✅ Augmented dataset saved as {output_file}")
print(f"📊 Final dataset shape: {df_final.shape}")
print("🔝 Sample rows:")
print(df_final.sample(5))
