import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Generator function to yield batches of data
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data.iloc[i:i+batch_size]

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
def semantic_search(input_text, data_batch):
    matched_rows = []

    # Tokenize input text and extract features
    input_encoded = tokenize_text(input_text)
    input_features = extract_features(input_encoded)

    # Tokenize and extract features for all descriptions in the dataset
    descriptions = data_batch['description'].tolist()
    description_encoded = tokenize_text(descriptions)
    description_features = extract_features(description_encoded)

    # Compute cosine similarity between input and description features
    similarity_scores = cosine_similarity(input_features, description_features)

    # Find rows with non-zero similarity scores
    for i, sim_score in enumerate(similarity_scores.flatten()):
        if sim_score > 0:
            matched_rows.append(data_batch.iloc[i])

    return matched_rows

# Main function
def main():
    # Load data from Excel file in chunks
    excel_file = 'your_excel_file.xlsx'
    batch_size = 16
    for data_batch in batch_generator(pd.read_excel(excel_file, chunksize=batch_size), batch_size):
        # Input text
        input_text = "Your input text goes here."

        # Perform semantic search on the current batch
        matched_rows = semantic_search(input_text, data_batch)

        # Display matched rows if found
        if matched_rows:
            print("Match found for input text in current batch:")
            for row in matched_rows:
                print(row)
        else:
            print("No match found for input text in current batch.")

if __name__ == "__main__":
    main()
