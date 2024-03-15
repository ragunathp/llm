import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Load data from Excel file
def load_data(excel_file):
    data = pd.read_excel(excel_file)
    return data

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

# Semantic search: Find rows in Excel data matching input text
def semantic_search(input_text, data):
    # Tokenize input text and extract features
    input_encoded = tokenize_text(input_text)
    input_features = extract_features(input_encoded)

    # Tokenize and extract features for all descriptions in the dataset
    descriptions = data['description'].tolist()
    description_encoded = tokenize_text(descriptions)
    description_features = extract_features(description_encoded)

    # Compute cosine similarity between input and description features
    similarity_scores = cosine_similarity(input_features, description_features)

    # Find rows with highest similarity scores
    max_similarity_index = similarity_scores.argmax()
    max_similarity = similarity_scores[max_similarity_index][0]

    # Display all columns for the row with highest similarity
    if max_similarity > 0:
        matched_row = data.iloc[max_similarity_index]
        return matched_row
    else:
        return None

# Main function
def main():
    # Load data from Excel file
    excel_file = 'your_excel_file.xlsx'
    data = load_data(excel_file)

    # Input text
    input_text = "Your input text goes here."

    # Perform semantic search
    matched_row = semantic_search(input_text, data)

    # Display matched row if found
    if matched_row is not None:
        print("Match found for input text:")
        print(matched_row)
    else:
        print("No match found for input text.")

if __name__ == "__main__":
    main()
