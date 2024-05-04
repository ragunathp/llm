import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load DistilBertTokenizer and DistilBertModel
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Read Excel file and extract a specific column with incident root cause text
def read_excel_file(excel_file, column_name):
    data = pd.read_excel(excel_file)
    incident_texts = data[column_name].dropna().tolist()
    return incident_texts

# Function to tokenize text and extract features using DistilBERT
def extract_features(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

# Function to summarize text based on the most representative sentences
def summarize_text(text_list, top_n=3):
    # Get sentence embeddings
    sentence_embeddings = extract_features(text_list)

    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(sentence_embeddings.numpy())
    
    # Choose the most representative sentences by their average similarity
    avg_similarities = similarity_matrix.mean(axis=1)
    top_indices = avg_similarities.argsort()[::-1][:top_n]  # Top N sentences with highest average similarity

    # Extract the most representative sentences
    top_sentences = [text_list[i] for i in top_indices]

    return top_sentences

# Function to extract keywords using TF-IDF
def extract_keywords(text_list, custom_stop_words=None):
    # Combine default stop words with custom ones if needed
    from sklearn.feature_extraction import text
    stop_words = text.ENGLISH_STOP_WORDS.union(custom_stop_words if custom_stop_words else set())
    
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=100)
    tfidf_matrix = vectorizer.fit_transform(text_list)
    
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    word_scores = dense.sum(axis=0).tolist()[0]
    keyword_score_pairs = sorted(zip(feature_names, word_scores), key=lambda x: x[1], reverse=True)

    top_keywords = keyword_score_pairs[:10]  # Top 10 keywords
    return top_keywords

# Example usage
excel_file = 'your_excel_file.xlsx'
column_name = 'incident_root_cause'

# Read the Excel file and extract incident root cause text
incident_texts = read_excel_file(excel_file, column_name)

# Summarize the text to find the top 3 representative sentences
top_sentences = summarize_text(incident_texts, top_n=3)

# Find common keywords while ignoring stop words
common_keywords = extract_keywords(incident_texts)

# Create a DataFrame to store the common keywords, their scores, and summarized sentences
summary_df = pd.DataFrame(common_keywords, columns=['Keyword', 'Score'])

# Add summarized sentences to the DataFrame
summary_df['Representative_Sentences'] = top_sentences[:len(summary_df)]

# Write the DataFrame to a new Excel file
output_excel_file = 'summary_with_keywords.xlsx'
summary_df.to_excel(output_excel_file, index=False)

print(f"Common keywords, scores, and representative sentences have been written to {output_excel_file}.")
