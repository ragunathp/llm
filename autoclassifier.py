import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load DistilBert tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained(model_name)

# Function to extract embeddings from incident root cause text
def extract_embeddings(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings
    return embeddings

# Function to cluster embeddings using K-means
def cluster_with_kmeans(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings.numpy())  # K-means expects numpy arrays
    return kmeans.labels_

# Read the Excel file and extract the incident root cause text
incident_texts = pd.read_excel("your_excel_file.xlsx")["incident_root_cause"].dropna().astype(str).tolist()

# Extract embeddings using DistilBert
embeddings = extract_embeddings(incident_texts)

# Cluster incidents using K-means with a specified number of clusters
cluster_labels = cluster_with_kmeans(embeddings, n_clusters=5)

# Create a DataFrame with incident texts and their corresponding clusters
incident_clusters = pd.DataFrame({
    "Incident": incident_texts,
    "Cluster": cluster_labels,
})

# Write the clustered incidents to an Excel file
incident_clusters.to_excel("incident_clusters.xlsx", index=False)
