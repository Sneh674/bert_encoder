import re
import json

from datasets import load_dataset

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

# print(type(ds))
# print(len(ds["train"]["text"]))

# Example text
text = """
I like machine learning.
I like deep learning.
Machine learning is interesting.
"""

# Step 1: Tokenize text (lowercase and remove punctuation)
tokens = re.findall(r"\b\w+\b", text.lower())

# Step 2: Add special tokens
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

word_to_id = {token: idx for idx, token in enumerate(special_tokens)}

# Step 3: Build vocabulary
for word in tokens:
    if word not in word_to_id:
        word_to_id[word] = len(word_to_id)

num_sentences = len(ds["train"]["text"])
num_sentences //= 50
print("total sentences:", num_sentences)
for i in range(num_sentences):
    print(f"Processing sentence {i+1}/{num_sentences}")
    text = ds["train"]["text"][i]
    tokens = re.findall(r"\b\w+\b", text.lower())
    for word in tokens:
        if word not in word_to_id:
            word_to_id[word] = len(word_to_id)

# Step 4: Create reverse dictionary
id_to_word = {idx: word for word, idx in word_to_id.items()}

# Step 5: Convert text into word IDs
word_ids = [word_to_id.get(word, word_to_id["[UNK]"]) for word in tokens]

print("Vocabulary:", word_to_id)
print("Word IDs:", word_ids)

# Step 6: Save vocabulary and word IDs to JSON files
with open("word_to_id.json", "w") as f:
    json.dump(word_to_id, f)

with open("id_to_word.json", "w") as f:
    json.dump(id_to_word, f)

with open("word_ids.json", "w") as f:
    json.dump(word_ids, f)