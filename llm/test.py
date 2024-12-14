from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained weights
model.load_state_dict(torch.load('regulation_model.pt'))
model.to(device)

def predict_violation(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits).item()
    return "违规" if prediction == 1 else "合规"

# Test specific cases
test_cases = [
    "电气支（吊）架锈蚀",
    "支（吊）架未做防锈处理",
    "金属支架已做防腐处理",
    "没问题",
    "电气支吊架吊杆直径不满足规范要求"
]

print("\n预测结果:")
for text in test_cases:
    result = predict_violation(text)
    print(f"问题: {text}")
    print(f"预测: {result}\n")