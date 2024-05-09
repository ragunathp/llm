import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openpyxl
import os

# Load pre-trained DistilBert tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Function to extract embeddings for text using DistilBert
def extract_embeddings(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings
    return embeddings

# Function to perform semantic search based on a combined query
def semantic_search(query, incident_data):
    # Extract embeddings for the query
    query_embedding = extract_embeddings([query])

    # Extract embeddings for incident texts
    incident_texts = incident_data["incident_text"].tolist()
    incident_embeddings = extract_embeddings(incident_texts)

    # Calculate cosine similarity between the query embedding and each incident embedding
    similarity_scores = [
        cosine_similarity(query_embedding.numpy(), embedding.unsqueeze(0).numpy())[0][0]
        for embedding in incident_embeddings
    ]

    # Get the indices of the most similar incidents
    most_similar_indices = np.argsort(similarity_scores)[::-1]  # Sort in descending order

    # Retrieve the most similar incidents based on similarity scores
    num_top_results = 5  # Get the top 5 most similar incidents
    top_results = []
    for i in most_similar_indices[:num_top_results]:
        result = {
            "Incident Number": incident_data.loc[i, "incident_number"],
            "Incident Text": incident_texts[i],
            "Score": similarity_scores[i],
            "Criticality": incident_data.loc[i, "criticality"],
            "Category": incident_data.loc[i, "category"],
            "Cause Type": incident_data.loc[i, "cause_type"]
        }
        top_results.append(result)

    return top_results

# Read Excel file and extract required columns
excel_file = 'incident_data.xlsx'  # The source Excel file
incident_data = pd.read_excel(excel_file, engine='openpyxl')[["incident_number", "incident_text", "criticality", "category", "cause_type"]]

# Output Excel file for results
output_excel_file = 'query_results.xlsx'

# Ensure the output file has the correct structure
if not os.path.exists(output_excel_file):
    # Create a DataFrame with appropriate columns if the file does not exist
    pd.DataFrame(columns=["Incident Number", "Incident Text", "Score", "Criticality", "Category", "Cause Type"]).to_excel(output_excel_file, index=False)

# Loop for continuous interaction
while True:
    # Ask the user for a combined query (category, criticality, cause type, incident text)
    query = input("Please enter your query (e.g., 'criticality: high, category: network, cause type: hardware'): ")

    # Perform semantic search to find the most relevant incidents based on the query
    top_results = semantic_search(query, incident_data)

    # Display the most relevant incidents
    print("\nTop 5 incidents most relevant to your query:")
    for i, result in enumerate(top_results):
        print(f"{i + 1}. Incident {result['Incident Number']}: {result['Incident Text']} (Score: {result['Score']:.2f}, Criticality: {result['Criticality']}, Category: {result['Category']}, Cause Type: {result['Cause Type']})")

    # Append the results to the output Excel file
    existing_data = pd.read_excel(output_excel_file, engine='openpyxl')
    new_data = pd.DataFrame(top_results)
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    updated_data.to_excel(output_excel_file, index=False)

    # Ask if the user wants to continue or exit
    continue_prompt = input("\nWould you like to ask another question? (yes/no): ").strip().lower()
    
    if continue_prompt not in ["yes", "y"]:
        print("Thank you! Goodbye!")
        break
