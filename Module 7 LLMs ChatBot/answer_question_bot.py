# Import necessary libraries
from flask import Flask, request, render_template, jsonify
from PyPDF2 import PdfReader  # Updated import
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import openai
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

app = Flask(__name__)

os.environ['OPENAI_API_KEY'] = "OPEN_API_KEY"

def extract_text_from_pdf(file_path):
    text = ''
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file) 
        for page in pdf_reader.pages: 
            text += page.extract_text()
    return text

embeddings = OpenAIEmbeddings()
document_search = None

chain = load_qa_chain(OpenAI(), chain_type="stuff")

chat_history = []

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'document' in request.files:
        document = request.files['document']
        if document.filename != '':
            document.save('uploaded_document.pdf')

            raw_text = extract_text_from_pdf('uploaded_document.pdf')
            text_spliter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200, length_function=len)
            texts = text_spliter.split_text(raw_text)

            global document_search
            document_search = FAISS.from_texts(texts, embeddings)

            return "Document uploaded successfully."
    return "No document uploaded."

@app.route('/ask', methods=['POST'])
def ask_question():
    if document_search is not None:
        query = request.form['question']
        docs = document_search.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)

        chat_history.append({"user": query, "bot": answer})

        return answer
    return "Please upload a document first."

@app.route('/chat_history')
def show_chat_history():
    return render_template('chat_history.html', chat=chat_history)

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    return jsonify(chat_history)

html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Document QA</title>
    <!-- Include Bootstrap CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <!-- Custom CSS for styling -->
    <style>
        body {
            margin: 20px;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
        }
        h1, h2, h3 {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Upload a PDF Document</h1>
        <form method="POST" action="/upload" enctype="multipart/form-data">
            <input type="file" name="document" class="form-control mb-2">
            <button type="submit" class="btn btn-primary">Upload</button>
        </form>

        <h2>Ask a Question</h2>
        <form method="POST" action="/ask">
            <input type="text" name="question" placeholder="Ask a question..." class="form-control mb-2">
            <button type="submit" class="btn btn-success">Ask</button>
        </form>

        <h3>Answer:</h3>
        <p id="answer"></p>
    </div>

    <script>
        document.querySelector('form[action="/ask"]').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const response = await fetch('/ask', {
                method: 'POST',
                body: formData
            });
            const answer = await response.text();
            document.getElementById('answer').textContent = answer;
            // Fetch the updated chat history after each interaction
            fetch('/get_chat_history')
                .then(response => response.json())
                .then(data => {
                    // Update the chat history section with the latest chat
                    const chatHistory = document.getElementById('chat_history');
                    chatHistory.innerHTML = '';
                    data.forEach(entry => {
                        const userMessage = entry.user;
                        const botMessage = entry.bot;
                        chatHistory.innerHTML += `<p><strong>User:</strong> ${userMessage}</p><p><strong>Bot:</strong> ${botMessage}</p><br>`;
                    });
                });
        });
    </script>
    <h2>Chat History</h2>
    <div id="chat_history"></div>
</body>
</html>
'''

history_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Chat History</title>
</head>
<body>
    <h1>Chat History</h1>
    <div id="chat_history">
        {% for entry in chat %}
            <p><strong>User:</strong> {{ entry['user'] }}</p>
            <p><strong>Bot:</strong> {{ entry['bot'] }}</p>
            <br>
        {% endfor %}
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    return html_template

if __name__ == '__main__':
    app.run(debug=True)
