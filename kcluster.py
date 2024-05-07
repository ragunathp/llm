import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Load pre-trained DistilBert tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Function to extract embeddings for text using DistilBert
def extract_embeddings(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no-grad():
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
            if any(keyword in text_lower for keyword in keywords):
                cluster_mapping[cluster_name].append((text, cluster_label))
                break

    return cluster_labels, cluster_mapping

# Function to visualize clusters with PCA and a legend
def visualize_clusters(embeddings, cluster_labels, cluster_mapping):
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    reduced_data = pca.fit_transform(embeddings.numpy())
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)

    # Add a colorbar with cluster numbers
    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Cluster Number")

    # Add a legend with keyword cluster names
    unique_clusters = set(cluster_labels)
    for cluster_label in unique_clusters:
        cluster_name = next(
            key
            for key, incidents in cluster_mapping.items()
            if any(item[1] == cluster_label for item in incidents)
        )
        plt.scatter([], [], c=plt.cm.viridis(cluster_label / (n_clusters - 1)), label=cluster_name)

    plt.title("PCA Visualization with Keyword Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    # Show the legend with the cluster names
    plt.legend(title="Keyword Clusters")
    plt.show()

# Predefined keyword clusters with associated keywords
keyword_clusters = {
    "Network Issues": ["network", "connection", "latency", "bandwidth", "packet loss"],
    "Memory Issues": ["memory", "heap", "ram", "out of memory", "memory leak"],
    "Database Issues": ["database", "sql", "query", "oracle", "db2", "mysql", "postgresql"],
    "Configuration Errors": ["configuration", "config", "setup", "settings"],
    "Human Errors": ["human error", "manual error", "mistake", "operator error"],
    "Hardware Failures": ["cpu", "processor", "core", "hardware", "motherboard"],
    "Storage Issues": ["storage", "disk", "ssd", "hdd", "capacity", "space"],
    "Firewall Issues": ["firewall", "rule", "access", "port"],
    "Load Balancer Issues": ["load balancer", "balancing", "lb", "nginx", "haproxy"],
    "TLS Certificate Issues": ["tls", "ssl", "certificate", "expiry", "renewal"],
    "Security Breaches": ["breach", "hack", "security", "unauthorized access", "compromise"],
    "Business Continuity": ["bcp", "business continuity", "disaster recovery", "backup"],
    "Patching Servers": ["patching", "update", "server update", "security patch"],
    "Restart Issues": ["restart", "reboot", "boot", "shutdown"],
    "Failover": ["failover", "redundancy", "high availability"],
}

# Read Excel file and extract incident root cause text
incident_texts = pd.read_excel("your_excel_file.xlsx")["incident_root_cause"].dropna().astype(str).tolist()

# Extract embeddings using DistilBert
embeddings = extract_embeddings(incident_texts)

# Cluster incidents based on embeddings and broader keyword clusters
cluster_labels, cluster_mapping = cluster_incidents_by_keywords(incident_texts, embeddings, keyword_clusters)

# Visualize clusters with PCA and the correct legend
visualize_clusters(embeddings, cluster_labels, cluster_mapping)

# Create a DataFrame to store incidents with clusters and associated keyword clusters
clustered_list = []
for cluster_name, incidents in cluster_mapping.items():
    for text, label in incidents:
        clustered_list.append({"Incident": text, "Cluster": label, "Keyword Cluster": cluster_name})

# Write the clustered incidents to an Excel file
output_excel_file = 'clustered_incidents_with_keyword_clusters.xlsx'
clustered_df.to_excel(output_excel_file, index=False)

print(f"Clustered incidents based on custom keyword clusters have been written to {output_excel_file}.")
