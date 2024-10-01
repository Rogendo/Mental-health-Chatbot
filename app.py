from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize conversation history
conversation_history = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_input = request.args.get('msg')
    return str(chatbot_response(user_input))

def chatbot_response(user_input):
    global conversation_history
    
    # Add user input to conversation history
    conversation_history.append(user_input)
    
    # Encode the entire conversation history
    input_ids = tokenizer.encode(" ".join(conversation_history), return_tensors="pt")
    
    # Generate a response
    output = model.generate(input_ids, max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    
    # Decode the response
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # Add bot response to conversation history
    conversation_history.append(response)
    
    # Keep only the last 5 exchanges to manage context length
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
    
    return response

if __name__ == "__main__":
    app.run()
