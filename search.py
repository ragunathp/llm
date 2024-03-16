import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity

# Initialize DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Main function
def main():
    # Load data from Excel file
    excel_file = 'your_excel_file.xlsx'
    input_text = "Your input text goes here."

    data = pd.read_excel(excel_file)

    # Extract header and data rows
    header = data.columns
    rows = data.values

    for i, row in enumerate(rows):
        # Skip header row
        if i == 0:
            continue
        
        # Process each row
        matched_rows_with_scores = semantic_search(input_text, header.tolist(), row.tolist())
        
        # Print matched rows and their similarity scores
        if matched_rows_with_scores:
            for matched_row, similarity_score in matched_rows_with_scores:
                print("Match found with similarity score:", similarity_score)
                print(matched_row)
                print()
                
                # Check if similarity score is greater than 90%
                if similarity_score > 0.9:
                    break  # Break out of the loop if score is greater than 90%
                
            if similarity_score > 0.9:
                break  # Break out of the outer loop if score is greater than 90%

    if not matched_rows_with_scores:
        print("No match found for input text.")

# Modified semantic_search function
def semantic_search(input_text, header, row):
    # Extract relevant information from the row and header
    category_index = header.index('category')  # Adjust column name as needed
    description_index = header.index('description')  # Adjust column name as needed

    category = row[category_index]
    description = row[description_index]

    # Perform semantic search based on category and description
    matched_rows_with_scores = perform_semantic_search(category, description, input_text)

    return matched_rows_with_scores

# Function to perform semantic search using DistilBERT embeddings and cosine similarity
def perform_semantic_search(category, description, input_text):
    # Tokenize input text, category, and description
    input_tokens = tokenizer.tokenize(input_text)
    category_tokens = tokenizer.tokenize(category)
    description_tokens = tokenizer.tokenize(description)

    # Encode tokens into IDs
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    category_ids = tokenizer.convert_tokens_to_ids(category_tokens)
    description_ids = tokenizer.convert_tokens_to_ids(description_tokens)

    # Convert IDs to PyTorch tensors
    input_ids = torch.tensor([input_ids])
    category_ids = torch.tensor([category_ids])
    description_ids = torch.tensor([description_ids])

    # Get DistilBERT embeddings for input text, category, and description
    with torch.no_grad():
        input_embeddings = model(input_ids)[0][:, 0, :]
        category_embeddings = model(category_ids)[0][:, 0, :]
        description_embeddings = model(description_ids)[0][:, 0, :]

    # Calculate cosine similarity between input text and category/description
    similarity_category = cosine_similarity(input_embeddings, category_embeddings)[0][0]
    similarity_description = cosine_similarity(input_embeddings, description_embeddings)[0][0]

    # Return similarity scores
    return [(f"Category: {category}, Similarity Score: {similarity_category}", similarity_category),
            (f"Description: {description}, Similarity Score: {similarity_description}", similarity_description)]

# Execute main function
main()
