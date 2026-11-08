import re

def read_txt(file_path="word_parsing/training.txt", max_seq_len=10):
    all_tokens = []

    # 1. Read the text file
    # file_path = "word_parsing/training.txt"

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Replace newlines with spaces so words across line breaks don't merge awkwardly
    text = text.replace("\n", ". ")

    # 2. Split text into sentences using regex (while punctuation is still intact)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # 3. Sequence through sentence by sentence, clean, and lowercase
    for index, sentence in enumerate(sentences, start=1):
        # Convert all letters to lower case
        sentence = sentence.lower()
        
        # Remove special characters (keep only lowercase letters a-z, numbers 0-9, and spaces)
        cleaned_sentence = re.sub(r"[^,a-z0-9\s]", "", sentence)
        cleaned_sentence = cleaned_sentence.replace(",", " [SEP]")
        
        # Clean up any extra spacing left behind
        cleaned_sentence = re.sub(r"\s+", " ", cleaned_sentence).strip()
        
        if cleaned_sentence:  # Skip empty strings
            # print(f"Sentence {index}: {cleaned_sentence}\n")
            sentence_token = cleaned_sentence.split()
            sentence_token = ["[CLS]"] + sentence_token 
            # + ["[SEP]"]
            # print(f"Tokens for Sentence {index}: {sentence_token}\n")
            for i in range(0, len(sentence_token), max_seq_len-1):
                if len(sentence_token) - i < max_seq_len-1:
                    difference = max_seq_len-1 - (len(sentence_token) - i)
                    # print(f"Length of sentence tokens: {len(sentence_token)}")
                    # print(f"Current index: {i}")
                    # print(f"difference = {difference}")
                    current_seq_tokens = sentence_token[i:i + max_seq_len-1] + difference*["[PAD]"] + ["[SEP]"]
                else:
                    current_seq_tokens=sentence_token[i:i + max_seq_len-1]+["[SEP]"]
                # print(f"Length of current sequence tokens: {len(current_seq_tokens)}")
                all_tokens.append(current_seq_tokens)

    # print(all_tokens)
    print("total sequences of tokens:",len(all_tokens))
    return all_tokens

# read_txt()
