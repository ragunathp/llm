import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader

# Load pre-trained DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Function to tokenize text and prepare input tensors
def tokenize_text(text_list, max_length=128):
    input_ids = []
    attention_masks = []

    for text in text_list:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,
                            max_length=max_length,
                            pad_to_max_length=True,
                            return_attention_mask=True,
                            return_tensors='pt',
                       )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

# Function to perform classification
def classify_texts(texts):
    # Tokenize and prepare input tensors
    input_ids, attention_masks = tokenize_text(texts)

    # Prepare DataLoader
    batch_size = 16
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = DataLoader(prediction_data, sampler=None, batch_size=batch_size)

    # Put model in evaluation mode
    model.eval()

    predictions = []

    # Predict
    for batch in prediction_sampler:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1]}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).tolist()
        predictions.extend(probabilities)

    return predictions

# Read Excel file
excel_file = 'your_excel_file.xlsx'
data = pd.read_excel(excel_file)

# Concatenate text from two columns
data['combined_text'] = data['column1'] + " " + data['column2']

# Assuming your Excel file has a column named 'combined_text' containing the concatenated text
texts = data['combined_text'].tolist()

# Classify texts
predictions = classify_texts(texts)

# Printing predictions
for text, prediction in zip(texts, predictions):
    print(f"Text: {text}")
    print(f"Prediction: {prediction}")
