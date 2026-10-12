from datasets import load_dataset

dataset = load_dataset("vicclab/fairy_tales", split="train")

output_path = "fairy_tales.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for row in dataset:
        story = row["text"].strip()

        if story:
            f.write(story)
            f.write("\n\n")

print(f"Saved {len(dataset)} stories to {output_path}")
