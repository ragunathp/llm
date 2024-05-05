import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text  # For customizing stop words

# Load pre-trained DistilBertTokenizer and DistilBertModel
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Function to read Excel and extract a specific column
def read_excel_file(excel_file, column_name):
    data = pd.read_excel(excel_file)
    incident_texts = data[column_name].dropna().astype(str).tolist()  # Convert to list of strings
    return incident_texts

# Function to extract keywords and map them to corresponding text snippets
def extract_keywords_and_text(text_list, custom_stop_words=None):
    # Define stop words
    default_stop_words = list(text.ENGLISH_STOP_WORDS)
    if custom_stop_words:
        stop_words = set(default_stop_words).union(set(custom_stop_words))
    else:
        stop_words = default_stop_words
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=100)
    tfidf_matrix = vectorizer.fit_transform(text_list)
    
    # Get the feature names and their scores
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    word_scores = dense.sum(axis=0).tolist()[0]
    keyword_score_pairs = sorted(zip(feature_names, word_scores), key=lambda x: x[1], reverse=True)

    # Create a mapping from keywords to corresponding text snippets
    keyword_to_text_map = {keyword: [] for keyword, _ in keyword_score_pairs}

    for text_snippet in text_list:
        for keyword in keyword_to_text_map.keys():
            if keyword in text_snippet.lower():  # Check if keyword is in the text snippet
                keyword_to_text_map[keyword].append(text_snippet)

    return keyword_score_pairs, keyword_to_text_map

# Example usage
excel_file = 'your_excel_file.xlsx'
column_name = 'incident_root_cause'

# Read the Excel file and extract incident root cause text
incident_texts = read_excel_file(excel_file, column_name)

# Extract keywords and map them to corresponding text snippets
custom_stop_words = ['incident', 'cause']  # Example custom stop words
common_keywords, keyword_to_text_map = extract_keywords_and_text(incident_texts, custom_stop_words)

# Create a DataFrame to store the common keywords and their corresponding text snippets
keyword_df = pd.DataFrame(common_keywords, columns=['Keyword', 'Score'])

# Add the corresponding text snippets to the DataFrame as a new column
keyword_df['Corresponding_Text'] = [', '.join(keyword_to_text_map[keyword]) for keyword in keyword_df['Keyword']]

# Write the DataFrame to a new Excel file
output_excel_file = 'keywords_with_texts.xlsx'
keyword_df.to_excel(output_excel_file, index=False)

print(f"Keywords and their corresponding text snippets have been written to {output_excel_file}.")
