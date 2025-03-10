import os
import openai
import faiss
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load AI model for FAISS
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from .docx
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    sentences = text.split(". ")  # Split into smaller chunks
    return sentences

# Create FAISS index
def create_faiss_index(sentences):
    embeddings = model.encode(sentences)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, sentences

# Search for relevant document chunks
def search_document(query, index, sentences):
    query_vector = model.encode([query])
    _, top_match = index.search(np.array(query_vector), 1)
    return sentences[top_match[0][0]] if top_match[0][0] >= 0 else "No relevant information found."

# Query ChatGPT with relevant text
def chat_with_docx(query, index, sentences):
    relevant_text = search_document(query, index, sentences)
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant answering based on a document."},
            {"role": "user", "content": f"Relevant document section:\n{relevant_text}\n\nUser question: {query}"}
        ]
    )
    return response.choices[0].message.content

# Flask Route: Upload Document
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    global index, sentences
    sentences = extract_text_from_docx(filepath)
    index, sentences = create_faiss_index(sentences)

    return jsonify({"message": "File uploaded and processed successfully"}), 200

# Flask Route: Ask Question
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    answer = chat_with_docx(query, index, sentences)
    return jsonify({"response": answer})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running!"})

# Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
