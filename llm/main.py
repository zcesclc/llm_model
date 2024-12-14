from transformers import AutoModel, AutoTokenizer

print("Loading model and tokenizer on CPU...")
# Load the model and tokenizer, forcing CPU usage
model = AutoModel.from_pretrained(
    "THUDM/chatglm-6b",
    trust_remote_code=True,
    device_map="cpu"  # Force CPU usage
).float()

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
print("Model and tokenizer loaded successfully.")

print("Processing query...")
# Run a simple query
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=None)
print("Query processed. Here's the response:\n")
print(response)
