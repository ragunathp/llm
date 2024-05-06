import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt

# Load pre-trained DistilBertTokenizer and DistilBertModel
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Function to tokenize text and extract embeddings using DistilBert
def extract_features(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings
    return embeddings

# Function to cluster incidents based on embeddings and map to broader keyword clusters
def cluster_incidents_by_keywords(incident_texts, embeddings, keyword_clusters):
    # Apply K-means clustering
    n_clusters = len(keyword_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings.numpy())  # K-means expects numpy arrays
    cluster_labels = kmeans.labels_

    # Create a mapping from broader keyword clusters to incidents
    cluster_mapping = defaultdict(list)

    for i, text in enumerate(incident_texts):
        cluster_label = cluster_labels[i]
        text_lower = text.lower()

        for cluster_name, keywords in keyword_clusters.items():
            # Check if any of the keywords in the keyword cluster are present in the text
            for keyword in keywords:
                if keyword in text_lower:
                    cluster_mapping[cluster_name].append((text, cluster_label))
                    break

    return cluster_labels, cluster_mapping

# Function to visualize clusters
def visualize_clusters(embeddings, labels):
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    reduced_data = pca.fit_transform(embeddings.numpy())
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title("Cluster Visualization")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

# Predefined keyword clusters with associated keywords
keyword_clusters = {
    "Network Issues": ["network", "connection", "latency", "bandwidth"],
    "Memory Issues": ["memory", "heap", "ram", "out of memory"],
    "Database Issues": ["database", "sql", "query", "oracle"],
    "Configuration Errors": ["configuration", "config", "setup"],
    "Human Errors": ["human error", "manual error"],
    "Business Continuity": ["bcp", "business continuity", "disaster recovery"]
}

# Example usage
excel_file = 'your_excel_file.xlsx'
column_name = 'incident_root_cause'

# Read the Excel file and extract incident root cause text
incident_texts = pd.read_excel(excel_file)[column_name].dropna().astype(str).tolist()

# Extract embeddings using DistilBert
embeddings = extract_features(incident_texts)

# Cluster incidents based on embeddings and broader keyword clusters
cluster_labels, cluster_mapping = cluster_incidents_by_keywords(incident_texts, embeddings, keyword_clusters)

# Visualize the clusters
visualize_clusters(embeddings, cluster_labels)

# Create a DataFrame to store the incidents with clusters and associated keywords
clustered_list = []
for cluster_name, incidents in cluster_mapping.items():
    for text, label in incidents:
        clustered_list.append({"Incident": text, "Cluster": label, "Keyword Cluster": cluster_name})

clustered_df = pd.DataFrame(clustered_list)

# Write the clustered incidents to an Excel file
output_excel_file = 'clustered_incidents_by_keyword_clusters.xlsx'
clustered_df.to_excel(output_excel_file, index=False)

print(f"Clustered incidents based on custom keyword clusters have been written to {output_excel_file}.")
