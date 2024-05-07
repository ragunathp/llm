import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity

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

# Function to match incidents to multiple keyword clusters based on a similarity threshold
def match_to_keyword_clusters(incident_texts, embeddings, keyword_clusters, threshold=0.7):
    # Create embeddings for keyword clusters
    keyword_cluster_embeddings = {}
    for cluster_name, keywords in keyword_clusters.items():
        keyword_embeddings = extract_embeddings(keywords)  # Embeddings for keywords
        # Take the mean embedding to represent the cluster
        keyword_cluster_embeddings[cluster_name] = torch.mean(keyword_embeddings, dim=0)

    # Match incidents to all keyword clusters with similarity above the threshold
    cluster_matches = []

    for incident_embedding, incident_text in zip(embeddings, incident_texts):
        similarities = [
            cosine_similarity(incident_embedding.unsqueeze(0).numpy(), cluster_embedding.unsqueeze(0).numpy())[0][0]
            for cluster_embedding in keyword_cluster_embeddings.values()
        ]
        
        matching_clusters = [
            list(keyword_clusters.keys())[i]
            for i, similarity in enumerate(similarities)
            if similarity >= threshold  # Only keep matches above the threshold
        ]
        
        # Add all matching clusters for the incident
        for cluster_name in matching_clusters:
            cluster_matches.append({"Incident": incident_text, "Cluster": cluster_name})

    return cluster_matches

# Predefined keyword clusters for various incident types
keyword_clusters = {
    "Network Issues": ["network", "connection", "latency", "bandwidth", "packet loss"],
    "Memory Issues": ["memory", "heap", "ram", "out of memory", "memory leak"],
    "Database Issues": ["database", "sql", "query", "oracle", "db2", "mysql", "postgresql"],
    "Configuration Errors": ["configuration", "config", "setup", "settings"],
    "Human Errors": ["human error", "manual error", "operator error", "human mistake"],
    "Business Continuity": ["bcp", "business continuity", "disaster recovery", "backup"],
}

# Example usage
excel_file = 'your_excel_file.xlsx'
column_name = 'incident_root_cause'

# Read the Excel file and extract incident root cause text
incident_texts = pd.read_excel(excel_file)[column_name].dropna().astype(str).tolist()

# Extract embeddings using DistilBert
embeddings = extract_embeddings(incident_texts)

# Match incidents to multiple keyword clusters based on a similarity threshold
threshold = 0.7  # Adjust the similarity threshold as needed
cluster_matches = match_to_keyword_clusters(incident_texts, embeddings, keyword_clusters, threshold=threshold)

# Create a DataFrame with incidents and their matching clusters
clustered_df = pd.DataFrame(cluster_matches)

# Write the results to an Excel file
output_excel_file = 'multiple_clustered_incidents.xlsx'
clustered_df.to_excel(output_excel_file, index=False)

print(f"Incidents with multiple cluster matches based on a similarity threshold have been written to {output_excel_file}.")
