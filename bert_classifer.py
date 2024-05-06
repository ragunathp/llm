import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline

# Load DistilBERT tokenizer and model for zero-shot classification
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("valhalla/distilbert-joint-mnli-stsb")

# Create a pipeline for zero-shot classification
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Read Excel file and extract a specific column with incident root cause text
def read_excel_file(excel_file, column_name):
    data = pd.read_excel(excel_file)
    incident_texts = data[column_name].dropna().astype(str).tolist()  # Convert to list of strings
    return incident_texts

# Predefined categories for classification
category_labels = ["Network error", "Memory", "Database", "MQ"]

# Function to classify incidents based on zero-shot classification
def classify_incidents(incident_texts, category_labels):
    classified_incidents = []

    for text in incident_texts:
        # Get classification results for each text
        result = classifier(text, candidate_labels=category_labels, multi_class=True)

        # Find the label with the highest score
        best_label = result["labels"][result["scores"].index(max(result["scores"]))]
        classified_incidents.append({"Text": text, "Category": best_label, "Score": max(result["scores"])})

    return classified_incidents

# Example usage
excel_file = 'your_excel_file.xlsx'
column_name = 'incident_root_cause'

# Read the Excel file and extract incident root cause text
incident_texts = read_excel_file(excel_file, column_name)

# Classify incidents based on predefined categories
classified_incidents = classify_incidents(incident_texts, category_labels)

# Create a DataFrame with classified incidents and their scores
classified_df = pd.DataFrame(classified_incidents)

# Write the classified incidents to an Excel file
output_excel_file = 'classified_incidents_distilbert.xlsx'
classified_df.to_excel(output_excel_file, index=False)

print(f"Classified incidents with DistilBERT have been written to {output_excel_file}.")
