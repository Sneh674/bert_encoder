from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

tokens = tokenizer(
    "The cat is sleeping",
    return_tensors="pt"
)

input_ids = tokens["input_ids"]

print(input_ids)