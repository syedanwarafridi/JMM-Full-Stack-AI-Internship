from flask import Flask, request, render_template
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = Flask(__name__)

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval() 
tokenizer = BertTokenizer.from_pretrained(model_name)

@app.route('/')
def home():
    return render_template('index.html', predicted_class='')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits

    predicted_class = torch.argmax(logits, dim=1).item()

    class_labels = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

    predicted_class_label = class_labels[predicted_class]

    return render_template('index.html', predicted_class_label=predicted_class_label)

if __name__ == '__main__':
    app.run(debug=True)
