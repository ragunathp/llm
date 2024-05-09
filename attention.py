import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize
from nltk.chunk import RegexpParser
import openpyxl
import os

# Ensure necessary NLTK components are downloaded
nltk.download('punkt')  # For word tokenization
nltk.download('averaged_perceptron_tagger')  # For POS tagging

# Function to extract attention words from incident text
def extract_attention_words(text):
    # Tokenize and POS tag the text
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    pos_tags = pos_tag(tokens)  # Part-of-Speech tagging
    
    # Define a pattern to identify noun phrases
    grammar = "NP: {<DT>?<JJ>*<NN.*>+}"  # A simple noun phrase pattern
    parser = RegexpParser(grammar)  # Create a parser with this grammar
    
    # Extract noun phrases
    parsed = parser.parse(pos_tags)
    attention_words = []

    # Traverse the parsed tree to extract noun phrases
    for subtree in parsed.subtrees(filter=lambda t: t.label() == 'NP'):
        # Join words in the noun phrase
        noun_phrase = " ".join(word for word, tag in subtree.leaves())
        attention_words.append(noun_phrase)
    
    return attention_words

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
