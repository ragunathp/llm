import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load DistilBert tokenizer and model
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Function to tokenize text and extract embeddings using DistilBert
def extract_embeddings(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings
    return embeddings

# Function to match text to keyword clusters based on cosine similarity
def match_to_keyword_cluster(incident_texts, keyword_clusters):
    incident_embeddings = extract_embeddings(incident_texts)  # Extract embeddings for incident texts
    
    # Create embeddings for keyword clusters
    keyword_cluster_embeddings = {}
    for cluster_name, keywords in keyword_clusters.items():
        keyword_embeddings = extract_embeddings(keywords)  # Embeddings for keywords
        # Take the mean embedding to represent the cluster
        keyword_cluster_embeddings[cluster_name] = torch.mean(keyword_embeddings, dim=0)

    # Match incidents to the closest keyword cluster based on cosine similarity
    cluster_matches = []
    for incident_embedding, incident_text in zip(incident_embeddings, incident_texts):
        similarities = [
            cosine_similarity(incident_embedding.unsqueeze(0).numpy(), cluster_embedding.unsqueeze(0).numpy())[0][0]
            for cluster_embedding in keyword_cluster_embeddings.values()
        ]
        best_cluster = list(keyword_clusters.keys())[np.argmax(similarities)]  # Find the closest cluster
        cluster_matches.append({"Incident": incident_text, "Cluster": best_cluster})

    return cluster_matches

# Example predefined keyword clusters
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

# Match incidents to keyword clusters based on cosine similarity
cluster_matches = match_to_keyword_cluster(incident_texts, keyword_clusters)

# Create a DataFrame with the incident text and the corresponding cluster
clustered_df = pd.DataFrame(cluster_matches)

# Write the clustered incidents to an Excel file
output_excel_file = 'clustered_incidents_with_cosine_similarity.xlsx'
clustered_df.to_excel(output_excel_file, index=False)

print(f"Clustered incidents based on cosine similarity with keyword clusters have been written to {output_excel_file}.")
