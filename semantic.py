import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Tokenize input text using DistilBERT tokenizer
def tokenize_text(text_list, max_length=128):
    encoded_dict = tokenizer(text_list, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return encoded_dict

# Extract features from tokenized input using DistilBERT model
def extract_features(encoded_dict):
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    last_hidden_state = outputs.last_hidden_state
    features = last_hidden_state[:, 0, :]
    return features

# Semantic search: Find top 5 rows in chunk of data matching input text
def semantic_search(input_text, chunk):
    # Tokenize input text and extract features
    input_encoded = tokenize_text(input_text)
    input_features = extract_features(input_encoded)

    # Tokenize and extract features for all descriptions in the chunk
    descriptions = chunk['description'].tolist()
    description_encoded = tokenize_text(descriptions)
    description_features = extract_features(description_encoded)

    # Compute cosine similarity between input and description features
    similarity_scores = cosine_similarity(input_features, description_features)

    # Sort similarity scores in descending order and get indices of top 5 matches
    top_indices = similarity_scores.argsort(axis=None)[-5:][::-1]

    # Extract top 5 matching rows and their similarity scores
    top_matches = [(chunk.iloc[i], similarity_scores.item(i)) for i in top_indices]

    return top_matches


# Main function
def main():
    # Load data from Excel file in chunks
    excel_file = 'your_excel_file.xlsx'
    chunksize = 1000  # Adjust chunk size as needed
    input_text = "Your input text goes here."
    found_match = False

    with pd.ExcelFile(excel_file) as xls:
        sheet_names = xls.sheet_names

        for sheet_name in sheet_names:
            for chunk in pd.read_excel(xls, sheet_name=sheet_name, chunksize=chunksize):
                matched_row = semantic_search(input_text, chunk)
                if matched_row is not None:
                    found_match = True
                    print("Match found for input text:")
                    print(matched_row)
                    break
            if found_match:
                break

    if not found_match:
        print("No match found for input text.")

if __name__ == "__main__":
    main()
