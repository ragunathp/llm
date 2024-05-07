import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline

# Load DistilBert tokenizer and zero-shot classification model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("valhalla/distilbert-joint-mnli-stsb")

# Create a pipeline for zero-shot classification
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Read the Excel file and extract the incident root cause text
incident_texts = pd.read_excel("your_excel_file.xlsx")["incident_root_cause"].dropna().astype(str).tolist()

# Define predefined categories for classification
category_labels = [
    "Network Issues",
    "Memory Issues",
    "Database Issues",
    "Firewall Issues",
    "Load Balancer Issues",
    "TLS Certificate Issues",
    "Security Breaches",
    "Business Continuity",
    "Patching Servers",
    "Restart Issues",
    "Failover",
]

# Function to classify incidents using zero-shot classification
def classify_incidents(incident_texts, category_labels):
    classified_incidents = []

    for text in incident_texts:
        result = classifier(text, candidate_labels=category_labels, multi_class=True)
        best_label = result["labels"][result["scores"].index(max(result["scores"]))]
        best_score = max(result["scores"])

        classified_incidents.append({"Incident": text, "Category": best_label, "Score": best_score})

    return classified_incidents

# Classify incidents based on predefined categories
classified_incidents = classify_incidents(incident_texts, category_labels)

# Create a DataFrame with classified incidents and their scores
classified_df = pd.DataFrame(classified_incidents)

# Write the classified incidents to an Excel file
classified_df.to_excel("classified_incidents.xlsx", index=False)
