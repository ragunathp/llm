import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load pre-trained DistilBertTokenizer and DistilBertModel
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Read Excel file and extract a specific column with incident root cause text
def read_excel_file(excel_file, column_name):
    data = pd.read_excel(excel_file)
    incident_texts = data[column_name].dropna().astype(str).tolist()  # Convert to list of strings
    return incident_texts

# Function to tokenize text and extract features using DistilBert
def extract_features(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings
    return embeddings

# Function to cluster text based on DistilBert embeddings
def cluster_texts(embeddings, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)  # Set random_state for reproducibility
    kmeans.fit(embeddings.numpy())  # KMeans expects numpy arrays
    return kmeans.labels_

# Function to visualize clusters using PCA and t-SNE
def visualize_clusters(embeddings, labels):
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    reduced_data = pca.fit_transform(embeddings.numpy())
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title("Cluster Visualization with PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

# Example usage
excel_file = 'your_excel_file.xlsx'
column_name = 'incident_root_cause'

# Read the Excel file and extract incident root cause text
incident_texts = read_excel_file(excel_file, column_name)

# Extract embeddings using DistilBert
embeddings = extract_features(incident_texts)

# Cluster the embeddings to classify similar incidents
n_clusters = 3  # Adjust as needed
labels = cluster_texts(embeddings, n_clusters=n_clusters)

# Visualize the clusters
visualize_clusters(embeddings, labels)

# Create a DataFrame with text and their corresponding clusters
clustered_df = pd.DataFrame({"Text": incident_texts, "Cluster": labels})

# Write the clustered text to an Excel file
output_excel_file = 'clustered_incidents.xlsx'
clustered_df.to_excel(output_excel_file, index=False)

print(f"Clustered incidents have been written to {output_excel_file}.")
