from transformers import AutoModel, AutoTokenizer

print("Loading model and tokenizer on CPU...")
# Load the model without specifying device_map
model = AutoModel.from_pretrained(
    "THUDM/chatglm-6b",
    trust_remote_code=True
).float()  # Force full precision for CPU

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
print("Model and tokenizer loaded successfully.")

print("Processing query...")
# Run a simple query
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=None)
print("Query processed. Here's the response:\n")
print(response)
