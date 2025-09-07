from transformers import pipeline

paraphraser = pipeline("text2text-generation", model="prithivida/parrot_paraphraser_on_T5")


test_text = "I barely sleep at night and wake up often."
results = paraphraser(f"paraphrase: {test_text}", num_return_sequences=2, num_beams=4)

print("🔄 Paraphrase results:")
for r in results:
    print("-", r["generated_text"])
