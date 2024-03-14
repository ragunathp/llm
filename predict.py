import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load pre-trained DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Function to tokenize text and prepare input tensors
def tokenize_text(text_list, max_length=None):
    input_ids = []
    attention_masks = []

    max_len = 0  # Track the maximum length of tokenized sequences

    for text in text_list:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,
                            max_length=max_length,
                            pad_to_max_length=True,
                            return_attention_mask=True,
                            return_tensors='pt',
                       )
        
        # Update max_len if the tokenized sequence length is longer
        max_len = max(max_len, encoded_dict['input_ids'].size(1))

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    # Pad sequences to the maximum length
    for i in range(len(input_ids)):
        padding_length = max_len - input_ids[i].size(1)
        input_ids[i] = torch.cat([input_ids[i], torch.zeros((1, padding_length), dtype=torch.long)], dim=1)
        attention_masks[i] = torch.cat([attention_masks[i], torch.zeros((1, padding_length), dtype=torch.long)], dim=1)
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

# Function to train the model
def train_model(train_dataloader, val_dataloader, epochs=3, learning_rate=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)

        model.eval()
        val_loss = 0
        val_preds = []

        for batch in val_dataloader:
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_masks, labels=labels)

            val_loss += outputs.loss.item()
            logits = outputs.logits
            val_preds.extend(torch.argmax(logits, dim=1).tolist())

        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = accuracy_score(val_targets, val_preds)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Training loss: {avg_train_loss}')
        print(f'Validation loss: {avg_val_loss}')
        print(f'Validation accuracy: {val_accuracy}')

# Read Excel file
excel_file = 'your_excel_file.xlsx'
data = pd.read_excel(excel_file)

# Assuming your Excel file has columns named 'category', 'description', and 'ID'
X = data[['category', 'description']]
y = data['ID']

# Tokenize input texts
input_ids, attention_masks = tokenize_text(X['description'].tolist())

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(input_ids, y, test_size=0.1, random_state=42)

# Convert labels to tensor
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_val = torch.tensor(y_val.values, dtype=torch.long)

# Prepare DataLoader for training and validation sets
batch_size = 16
train_data = TensorDataset(X_train, attention_masks[:len(X_train)], y_train)
train_sampler = DataLoader(train_data, sampler=None, batch_size=batch_size)
val_data = TensorDataset(X_val, attention_masks[len(X_train):], y_val)
val_sampler = DataLoader(val_data, sampler=None, batch_size=batch_size)


# Determine device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
train_model(train_sampler, val_sampler)

# Now, let's use the trained model to predict the ID based on a sample input text
sample_text = "This is a sample description text."
sample_category = "Sample Category"

# Tokenize the sample input text
sample_input_ids, sample_attention_masks = tokenize_text([sample_text])

# Convert sample input to PyTorch tensors
sample_input_ids = sample_input_ids.to(device)
sample_attention_masks = sample_attention_masks.to(device)

# Use the trained model to predict the ID
with torch.no_grad():
    outputs = model(sample_input_ids, attention_mask=sample_attention_masks)

predicted_id = torch.argmax(outputs.logits, dim=1).item()

print(f"The predicted ID for the sample input text is: {predicted_id}")
