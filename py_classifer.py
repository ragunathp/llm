import pandas as pd
import re

# Function to classify incidents based on predefined keywords
def classify_incidents(incident_texts, category_keywords):
    # Create a DataFrame to store text and their corresponding categories
    incident_df = pd.DataFrame({"Text": incident_texts, "Category": ["Unclassified"] * len(incident_texts)})

    # Loop through the incident texts to classify based on keywords
    for i, text in enumerate(incident_texts):
        text_lower = text.lower()
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    incident_df.at[i, "Category"] = category
                    break

    return incident_df

# Example usage
excel_file = 'your_excel_file.xlsx'
column_name = 'incident_root_cause'

# Read the Excel file and extract incident root cause text
incident_texts = pd.read_excel(excel_file)[column_name].dropna().astype(str).tolist()

# Predefined categories and their associated keywords
category_keywords = {
    "Network error": ["network", "connection", "latency"],
    "Memory": ["memory", "heap", "out of memory"],
    "Database": ["database", "sql", "query"],
    "MQ": ["message queue", "mq", "queue"]
}

# Classify incidents based on predefined categories
classified_df = classify_incidents(incident_texts, category_keywords)

# Write the classified text to an Excel file
output_excel_file = 'classified_incidents.xlsx'
classified_df.to_excel(output_excel_file, index=False)

print(f"Classified incidents have been written to {output_excel_file}.")
