from datasets import load_dataset

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

print(type(ds))
print(len(ds["train"]["text"][1]))