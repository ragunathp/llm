import pandas as pd
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

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

# Function to match incidents to keyword clusters based on a similarity threshold
def match_to_keyword_clusters(incident_texts, embeddings, keyword_clusters, threshold=0.7):
    # Create embeddings for keyword clusters
    keyword_cluster_embeddings = {}
    for cluster_name, keywords in keyword_clusters.items():
        keyword_embeddings = extract_embeddings(keywords)  # Embeddings for keywords
        keyword_cluster_embeddings[cluster_name] = torch.mean(keyword_embeddings, dim=0)  # Mean embedding for cluster

    # Match incidents to all keyword clusters with similarity above the threshold
    cluster_mapping = defaultdict(list)
    unclassified = []

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
        
        if matching_clusters:
            for cluster_name in matching_clusters:
                cluster_mapping[cluster_name].append(incident_text)  # Add to multiple clusters
        else:
            unclassified.append(incident_text)  # No match above the threshold

    return cluster_mapping, unclassified

# Predefined keyword clusters for various incident types
keyword_clusters = {
    "Network Issues": ["network", "connection", "latency", "bandwidth", "packet loss"],
    "Memory Issues": ["memory", "heap", "ram", "out of memory", "memory leak"],
    "Database Issues": ["database", "sql", "query", "oracle", "db2", "mysql", "postgresql"],
    "Configuration Errors": ["configuration", "config", "setup", "settings"],
    "Human Errors": ["human error", "manual error", "operator error", "human mistake"],
    "Business Continuity": ["bcp", "business continuity", "disaster recovery", "backup"],
}

# Read Excel file and extract incident root cause text
incident_texts = pd.read_excel("your_excel_file.xlsx")["incident_root_cause"].dropna().astype(str).tolist()

# Extract embeddings using DistilBert
embeddings = extract_embeddings(incident_texts)

# Match incidents to multiple keyword clusters with a similarity threshold
threshold = 0.7  # Adjust the similarity threshold as needed
cluster_mapping, unclassified_incidents = match_to_keyword_clusters(incident_texts, embeddings, keyword_clusters, threshold=threshold)

# Create a DataFrame with incidents and their corresponding clusters
clustered_list = []
for cluster_name, incidents in cluster_mapping.items():
    for incident in incidents:
        clustered_list.append({"Incident": incident, "Cluster": cluster_name})

# Add unclassified incidents to the DataFrame
for incident in unclassified_incidents:
    clustered_list.append({"Incident": incident, "Cluster": "Unclassified"})

clustered_df = pd.DataFrame(clustered_list)

# Write the clustered incidents to an Excel file
output_excel_file = 'clustered_incidents_with_unclassified.xlsx'
clustered_df.to_excel(output_excel_file, index=False)

print(f"Clustered incidents with multiple cluster matches and unclassified category have been written to {output_excel_file}.")
