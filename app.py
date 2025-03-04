#Imports
from flask import Flask, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
import torch

# Load model from Hugging Face
MODEL_NAME = "ssdaimari44/bodo-mt"
model = MarianMTModel.from_pretrained(MODEL_NAME)
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)

# Initialize Flask app
app = Flask(__name__)


# Translation function
def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated_tokens = model.generate(**inputs)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)


# Define API route
@app.route("/translate", methods=["POST"])
def translate_text():
    data = request.get_json()
    if "text" not in data:
        return jsonify({"error": "Missing 'text' parameter"}), 400

    translated_text = translate(data["text"])
    return jsonify({"input": data["text"], "translation": translated_text})


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
