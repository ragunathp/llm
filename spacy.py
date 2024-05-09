import pandas as pd
import spacy
import openpyxl
import os

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Function to extract noun phrases (attention words) from text using SpaCy
def extract_attention_words(text):
    doc = nlp(text.lower())  # Process the text with SpaCy
    # Extract noun phrases
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    return noun_phrases

# Read Excel file with incident texts
excel_file = 'incident_data.xlsx'  # Replace with your Excel file
incident_data = pd.read_excel(excel_file, engine='openpyxl')

# Output Excel file for results
output_excel_file = 'attention_words_from_incidents.xlsx'

# Ensure the output file has the correct structure
if not os.path.exists(output_excel_file):
    # Create an empty DataFrame with appropriate columns
    pd.DataFrame(columns=["Incident Number", "Incident Text", "Attention Words"]).to_excel(output_excel_file, index=False)

# Extract attention words from incident texts and write to Excel
results = []
for idx, row in incident_data.iterrows():
    incident_number = row.get("incident_number", "Unknown")
    incident_text = row.get("incident_text", "")

    # Extract attention words from the incident text using SpaCy
    attention_words = extract_attention_words(incident_text)

    # Store the incident details and the extracted attention words
    results.append({
        "Incident Number": incident_number,
        "Incident Text": incident_text,
        "Attention Words": ", ".join(attention_words)  # Join all noun phrases
    })

# Write the results to the Excel output file
results_df = pd.DataFrame(results)
results_df.to_excel(output_excel_file, index=False)

print(f"Extracted attention words from incidents have been written to '{output_excel_file}'.")


------0

import pandas as pd
import re  # For basic tokenization with regex
from collections import Counter
import openpyxl
import os

# Function to tokenize text without punkt
def basic_tokenize(text):
    # Split text into words using regex (split on whitespace and punctuation)
    tokens = re.findall(r'\w+', text.lower())  # \w+ matches sequences of alphanumeric characters
    return tokens

# Function to extract attention words from incident text (using simple noun phrases)
def extract_attention_words(text):
    # Tokenize the text
    tokens = basic_tokenize(text)
    
    # Use common word patterns to extract noun phrases
    # For simplicity, consider sequences of nouns or adjectives + nouns as noun phrases
    noun_phrases = []
    current_phrase = []
    
    for token in tokens:
        # Assuming nouns and adjectives can be part of a noun phrase
        # This is a simplified approach without explicit POS tagging
        if re.match(r'[a-z]+', token):  # Match only alphabetical characters
            current_phrase.append(token)
        else:
            if current_phrase:
                noun_phrases.append(' '.join(current_phrase))
                current_phrase = []
    
    # Add any remaining phrases
    if current_phrase:
        noun_phrases.append(' '.join(current_phrase))
    
    return noun_phrases

# Read Excel file with incident texts
excel_file = 'incident_data.xlsx'
incident_data = pd.read_excel(excel_file, engine='openpyxl')

# Output Excel file for results
output_excel_file = 'attention_words_from_incidents.xlsx'

# Ensure the output file has the correct structure
if not os.path.exists(output_excel_file):
    # Create an empty DataFrame with appropriate columns
    pd.DataFrame(columns=["Incident Number", "Incident Text", "Attention Words"]).to_excel(output_excel_file, index=False)

# Extract attention words from incident texts and write to Excel
results = []
for idx, row in incident_data.iterrows():
    incident_number = row.get("incident_number", "Unknown")
    incident_text = row.get("incident_text", "")

    # Extract attention words from the incident text
    attention_words = extract_attention_words(incident_text)

    # Store the incident details and the extracted attention words
    results.append({
        "Incident Number": incident_number,
        "Incident Text": incident_text,
        "Attention Words": ", ".join(attention_words)  # Join all noun phrases
    })

# Write the results to the Excel output file
results_df = pd.DataFrame(results)
results_df.to_excel(output_excel_file, index=False)

print(f"Extracted attention words from incidents have been written to '{output_excel_file}'.")

--------

import pandas as pd
import spacy
import re  # For detecting negative phrases
import os
import openpyxl

# Load SpaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Predefined list of common negative words
negative_words = ["not", "no", "never", "error", "fail", "failure", "problem", "issue"]

# Function to extract negative noun phrases from text
def extract_negative_phrases(text):
    doc = nlp(text.lower())  # Process the text with SpaCy
    
    # Extract noun phrases containing negative words
    negative_phrases = []
    for chunk in doc.noun_chunks:
        # Check if the noun phrase contains any negative word
        if any(neg_word in chunk.text for neg_word in negative_words):
            negative_phrases.append(chunk.text)  # Add the negative noun phrase
    
    return negative_phrases

# Read Excel file with incident texts
excel_file = 'incident_data.xlsx'  # Replace with your Excel file
incident_data = pd.read_excel(excel_file, engine='openpyxl')

# Output Excel file for results
output_excel_file = 'negative_phrases_from_incidents.xlsx'

# Ensure the output file has the correct structure
if not os.path.exists(output_excel_file):
    # Create an empty DataFrame with appropriate columns
    pd.DataFrame(columns=["Incident Number", "Incident Text", "Negative Phrases"]).to_excel(output_excel_file, index=False)

# Extract negative phrases from incident texts and write to Excel
results = []
for idx, row in incident_data.iterrows():
    incident_number = row.get("incident_number", "Unknown")
    incident_text = row.get("incident_text", "")

    # Extract negative noun phrases from the incident text
    negative_phrases = extract_negative_phrases(incident_text)

    if negative_phrases:
        results.append({
            "Incident Number": incident_number,
            "Incident Text": incident_text,
            "Negative Phrases": ", ".join(negative_phrases)  # Join all negative phrases
        })

# Write the results to the Excel output file
results_df = pd.DataFrame(results)
results_df.to_excel(output_excel_file, index=False)

print(f"Extracted negative phrases from incidents have been written to '{output_excel_file}'.")
----------------------------
import pandas as pd
import spacy
import re
import openpyxl
import os

# Load SpaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Comprehensive list of negative words, including all the categories provided
negative_words = [
    # General Negative Words for Production
    "bug", "defect", "flaw", "glitch", "malfunction", "crash", "freeze", "unresponsive", "slow", "bottleneck",
    "resource leak", "inconsistency", "deadlock", "instability", "outage", "downtime",
    
    # Specific Coding Errors
    "syntax error", "compile error", "runtime error", "unhandled exception", "stack overflow", "memory leak",
    "segmentation fault", "null pointer", "type mismatch", "array index out of bounds", "division by zero",
    "race condition", "infinite loop", "timeout", "data corruption", "buffer overflow",
    
    # Performance and Stability Issues
    "high latency", "throughput bottleneck", "resource contention", "memory exhaustion", "CPU overload",
    "memory fragmentation", "resource deadlock", "infinite recursion", "system crash", "service interruption",
    
    # Deployment and Environment Issues
    "environment mismatch", "incorrect environment variables", "missing dependency", "misconfigured environment",
    "incompatible library", "dependency conflict", "outdated version", "broken build", "failed deployment",
    "missing configuration", "deployment rollback",
    
    # Security and Authorization Issues
    "security vulnerability", "privilege escalation", "unauthorized access", "injection attack", "SQL injection",
    "cross-site scripting", "cross-site request forgery", "insecure code", "insecure storage", "insecure authentication",
    "insecure authorization",
    
    # Software Development Process Issues
    "unapproved change", "missing unit tests", "broken tests", "untested code", "insufficient testing",
    "missing documentation", "code smell", "code duplication", "code refactoring required", "code complexity",
    "code maintainability issue", "technical debt"
]

# Function to extract phrases containing negative words from incident text
def extract_negative_phrases(text):
    doc = nlp(text.lower())  # Process text with SpaCy
    
    # Extract noun phrases containing negative words
    negative_phrases = []
    for chunk in doc.noun_chunks:
        # Check if the noun phrase contains any negative word
        if any(neg_word in chunk.text for neg_word in negative_words):
            negative_phrases.append(chunk.text)  # Add the negative noun phrase
    
    return negative_phrases

# Read Excel file with incident texts
excel_file = 'incident_data.xlsx'  # Replace with your Excel file
incident_data = pd.read_excel(excel_file, engine='openpyxl')

# Output Excel file for results
output_excel_file = 'negative_phrases_from_incidents.xlsx'

# Ensure the output file has the correct structure
if not os.path.exists(output_excel_file):
    # Create an empty DataFrame with appropriate columns
    pd.DataFrame(columns=["Incident Number", "Incident Text", "Negative Phrases"]).to_excel(output_excel_file, index=False)

# Extract negative phrases from incident texts and write to Excel
results = []
for idx, row in incident_data.iterrows():
    incident_number = row.get("incident_number", "Unknown")
    incident_text = row.get("incident_text", "")

    # Extract negative phrases from the incident text using SpaCy
    negative_phrases = extract_negative_phrases(incident_text)

    if negative_phrases:
        results.append({
            "Incident Number": incident_number,
            "Incident Text": incident_text,
            "Negative Phrases": ", ".join(negative_phrases)  # Join all negative phrases
        })

# Write the results to the Excel output file
results_df = pd.DataFrame(results)
results_df.to_excel(output_excel_file, index=False)

print(f"Extracted negative phrases from incidents have been written to '{output_excel_file}'.")
