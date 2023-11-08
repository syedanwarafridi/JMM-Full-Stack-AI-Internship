#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
from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
import openai
import os

# Initialize Flask app
app = Flask(__name__)
bootstrap = Bootstrap(app)

# Set your OpenAI API key using an environment variable
os.environ["OPENAI_API_KEY"] = 'OPENAIKEY'

@app.route('/')
def chatbot():
    return render_template('chatbot.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']

    # Use the OpenAI GPT-3 API with the v1/completions endpoint and the "text-davinci-003" model
    response = openai.Completion.create(
        engine="text-davinci-003",  # Specify the desired model
        prompt=user_input,
        max_tokens=50,  # You can adjust the max_tokens for response length
    )

    chatbot_response = response.choices[0].text

    print(f"User Input: {user_input}")
    print(f"Chatbot Response: {chatbot_response}")

    return jsonify({'response': chatbot_response})

if __name__ == '__main__':
    app.run(debug=True)
