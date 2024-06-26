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

-----------------

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import openpyxl
import os

# Ensure NLTK's VADER sentiment analyzer is available
nltk.download('vader_lexicon')

# Initialize NLTK's sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Function to extract positive and negative words from text
def extract_sentiment_words(text):
    doc = nltk.word_tokenize(text.lower())  # Tokenize the text
    positive_words = []
    negative_words = []
    
    # Analyze sentiment for each token
    for token in doc:
        sentiment_score = sentiment_analyzer.polarity_scores(token)  # Get sentiment score
        if sentiment_score['compound'] > 0.05:  # Positive threshold
            positive_words.append(token)  # Add to positive words
        elif sentiment_score['compound'] < -0.05:  # Negative threshold
            negative_words.append(token)  # Add to negative words
    
    return positive_words, negative_words  # Return both positive and negative words

# Read Excel file with incident texts
excel_file = 'incident_data.xlsx'  # Replace with your Excel file
incident_data = pd.read_excel(excel_file, engine='openpyxl')

# Output Excel file for results
output_excel_file = 'sentiment_words_from_incidents.xlsx'

# Ensure the output file has the correct structure
if not os.path.exists(output_excel_file):
    # Create an empty DataFrame with appropriate columns
    pd.DataFrame(columns=["Incident Text", "Positive Words", "Negative Words"]).to_excel(output_excel_file, index=False)

# Extract positive and negative words from incident texts and write to Excel
results = []
for idx, row in incident_data.iterrows():
    incident_text = row.get("incident_text", "")

    # Extract positive and negative words from the incident text
    positive_words, negative_words = extract_sentiment_words(incident_text)

    # Concatenate the words into a single string
    positive_concat = ", ".join(set(positive_words))  # Unique positive words
    negative_concat = ", ".join(set(negative_words))  # Unique negative words

    results.append({
        "Incident Text": incident_text,
        "Positive Words": positive_concat,
        "Negative Words": negative_concat
    })

# Write the results to the Excel output file
results_df = pd.DataFrame(results)
results_df.to_excel(output_excel_file, index=False)

print(f"Sentiment words from incidents have been written to '{output_excel_file}'.")
------------------

import pandas as pd
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# Ensure NLTK's VADER sentiment analyzer is available
nltk.download('vader_lexicon')

# Load SpaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Initialize NLTK's sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment and classify text as positive, neutral, or negative
def classify_sentiment(text):
    sentiment_score = sentiment_analyzer.polarity_scores(text.lower())
    if sentiment_score['compound'] > 0.05:
        return "positive"
    elif sentiment_score['compound'] < -0.05:
        return "negative"
    else:
        return "neutral"

# Read Excel file with incident texts
excel_file = 'incident_data.xlsx'  # Replace with your Excel file
incident_data = pd.read_excel(excel_file, engine='openpyxl')

# Separate texts into positive and negative based on sentiment
positive_texts = []
negative_texts = []

for idx, row in incident_data.iterrows():
    incident_text = row.get("incident_text", "")
    sentiment_class = classify_sentiment(incident_text)

    if sentiment_class == "positive":
        positive_texts.append(incident_text)
    elif sentiment_class == "negative":
        negative_texts.append(incident_text)

# Function to generate a word cloud from a list of texts
def generate_word_cloud(texts, title):
    combined_texts = " ".join(texts)  # Combine all texts into a single string
    word_cloud = WordCloud(background_color='white', width=800, height=400).generate(combined_texts)

    plt.figure(figsize=(10, 6))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.show()

# Generate word clouds for positive and negative texts
if positive_texts:
    generate_word_cloud(positive_texts, "Positive Word Cloud")

if negative_texts:
    generate_word_cloud(negative_texts, "Negative Word Cloud")
    
    ---------------------
    import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import openpyxl
import os

# Set NLTK data path and load necessary resources
nltk.data.path.append("/your/custom/path")  # Set your custom data path
nltk.download("punkt", download_dir="/your/custom/path")  # Ensure "punkt" is available
nltk.download("vader_lexicon", download_dir="/your/custom/path")  # Ensure VADER lexicon is available

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Read Excel file with incident texts
excel_file = 'incident_data.xlsx'  # Replace with your Excel file
incident_data = pd.read_excel(excel_file, engine='openpyxl')

# Output Excel file for sentiment results
output_excel_file = 'sentiment_analysis_incidents.xlsx'

# Output text file for unique negative words
output_txt_file = 'unique_negative_words.txt'

# Ensure the Excel output file has the correct structure
if not os.path.exists(output_excel_file):
    # Create an empty DataFrame with appropriate columns
    pd.DataFrame(columns=["Incident Text", "Positive Words", "Negative Words"]).to_excel(output_excel_file, index=False)

# List to collect all unique negative words
all_negative_words = set()  # Using a set to ensure uniqueness

# Analyze sentiment and tokenize words
results = []
for idx, row in incident_data.iterrows():
    incident_text = row.get("incident_text", "")

    # Tokenize the text into words
    tokens = word_tokenize(incident_text)  # Should work if "punkt" is loaded

    # Analyze sentiment and extract positive and negative words
    positive_words = []
    negative_words = []

    for token in tokens:
        sentiment_score = analyzer.polarity_scores(token)  # Get sentiment score for each token
        if sentiment_score['compound'] > 0.05:
            positive_words.append(token)  # Positive words
        elif sentiment_score['compound'] < -0.05:
            negative_words.append(token)  # Negative words
            all_negative_words.add(token)  # Collect unique negative words

    # Concatenate positive and negative words for Excel output
    positive_concat = ", ".join(set(positive_words))
    negative_concat = ", ".join(set(negative_words))

    results.append({
        "Incident Text": incident_text,
        "Positive Words": positive_concat,
        "Negative Words": negative_concat
    })

# Write the results to the Excel output file
results_df = pd.DataFrame(results)
results_df.to_excel(output_excel_file, index=False)

# Write the unique negative words to a text file
with open(output_txt_file, 'w') as f:
    unique_negative_words = sorted(all_negative_words)  # Sort the unique negative words
    f.write('"' + '", "'.join(unique_negative_words) + '"')  # Write with double quotes and comma-separated

print(f"Sentiment analysis results written to '{output_excel_file}'.")
print(f"Unique negative words written to '{output_txt_file}'.")

