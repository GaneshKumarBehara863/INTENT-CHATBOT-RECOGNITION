import json
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the intent data from intent.json
with open('./Intent.json', 'r') as file:
    intents = json.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_intent', methods=['POST'])
def get_intent():
    user_question = request.json.get('user_question')

    # Initialize variables for tag and response
    intent_tag = None
    response = "I'm sorry, I couldn't understand your question."

    # Loop through the intents to find a match
    for intent in intents['intents']:
        if user_question in intent['patterns']:
            intent_tag = intent['intent_tag']
            response = intent['responses'][0]  # Use the first response

    return jsonify({
        "intent_tag": intent_tag,
        "response": response,
        "user_question": user_question
    })

if __name__ == '__main__':
    app.run(debug=True)
