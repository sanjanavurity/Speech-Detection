from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import enchant  # 🔥 add this
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Load model and tokenizer
model_path = "hate_speech_model"
tokenizer_path = "hate_speech_tokenizer"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set to evaluation mode

# Initialize Flask app
app = Flask(__name__)

# Initialize English dictionary for validation
english_dict = enchant.Dict("en_US")

# Define label mapping
label_map = {2: "Non-Hate Speech", 1: "Offensive Speech", 0: "Hate Speech"}

@app.route('/')
def home():
    return "Hello Flask"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # 🔥 Step 1: Validate input text
        cleaned_text = re.sub(r'[^A-Za-z\s]', '', text)  # remove non-alphabet
        words = cleaned_text.split()
        valid_word_count = sum(1 for word in words if english_dict.check(word))

        if valid_word_count < 1:
            return jsonify({'prediction': "Invalid Input-Please Enter Proper text"})  # 🔥 return Invalid if random text

        # Extended negation pattern override
        negation_keywords = [
            "don't hate", "do not hate", "not hate", "never hate",
            "didn't hate", "doesn't hate", "won't hate", "can't hate",
            "not racist", "never racist", "don't insult", "not insult",
            "not offensive", "not a terrorist", "not mean", "no hate",
            "not bullying", "not harmful", "not hurtful", "never offend",
            "not targeting", "don't target", "not discriminating"
        ]
        
        if any(phrase in text.lower() for phrase in negation_keywords):
            return jsonify({'prediction': "Non-Hate Speech"})

        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        return jsonify({'prediction': label_map[prediction]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate', methods=['GET'])
def evaluate_model():
    try:
        # Load your test dataset
        df = pd.read_csv("test_data.csv")  # Your file must have 'text' and 'label' columns

        texts = df["text"].tolist()
        true_labels = df["label"].tolist()
        predicted_labels = []

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()
                predicted_labels.append(prediction)

        # Generate confusion matrix and report
        cm = confusion_matrix(true_labels, predicted_labels)
        report = classification_report(true_labels, predicted_labels, target_names=["Hate", "Offensive", "Non-Hate"])

        return jsonify({
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
