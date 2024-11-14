from transformers import BertTokenizer, BertModel
import torch

# Load the pre-trained BERT tokenizer and model
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Ensure model runs on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to perform inference
def get_bert_embeddings(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the embeddings from the last hidden layer
    embeddings = outputs.last_hidden_state
    return embeddings

# Example usage
text = "你好，这是一个测试。"
embeddings = get_bert_embeddings(text)

# Print the shape of the embeddings
print(f"Embeddings shape: {embeddings.shape}")
