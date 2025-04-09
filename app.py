import os
import sys
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import tempfile

# Import the question answering modules
from pdf_text_extractor import extract_text_from_pdf
from query_llm import query_flash
from text_chunker import smart_chunk_spacy_advanced
from vector_db_utils import generate_embeddings, store_in_faiss, load_faiss_index, query_faiss_index

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://pdf-query-app.vercel.app", "http://localhost:3000"]
    }
})


def process_pdf_query(pdf_file, query_text, relevance_threshold=np.float32(0.6)):
    """
    Process PDF query using the question answering pipeline
    """
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        pdf_file.save(temp_file.name)
        pdf_path = temp_file.name

    try:
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_path)
        if not extracted_text:
            return {"error": "Failed to extract text from the PDF"}

        # Chunk the text
        chunks = smart_chunk_spacy_advanced(extracted_text)
        if not chunks:
            return {"error": "Failed to create meaningful text chunks"}

        # Generate embeddings
        embeddings = generate_embeddings(chunks)

        # Store and load FAISS index
        index = store_in_faiss(embeddings, db_file="vector_db_cosine.index")
        index = load_faiss_index("vector_db_cosine.index")

        # Query FAISS index
        results = query_faiss_index(query_text, index, chunks, top_k=5)
        
        # Adjust relevance threshold
        while len(results) > 0 and all(score <= relevance_threshold for _, score, _ in results):
            relevance_threshold *= 0.5

        # Filter results by relevance threshold
        context_chunks = [(text, dis, idx) for text, dis, idx in results if dis > relevance_threshold]

        # Query LLM
        answer_result = query_flash(query_text, context_chunks)

        # Clean up temporary file
        os.unlink(pdf_path)

        return {
            "answer": answer_result.get("answer", "No answer could be generated"),
            "relevant_context": [chunk[0] for chunk in context_chunks]
        }

    except Exception as e:
        # Clean up temporary file in case of error
        if 'pdf_path' in locals():
            try:
                os.unlink(pdf_path)
            except:
                pass
        return {"error": f"An error occurred: {str(e)}"}

@app.route("/api/query", methods=["POST"])
def query_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    if "question" not in request.form:
        return jsonify({"error": "No question provided"}), 400

    file = request.files["file"]
    question = request.form["question"]

    # Process the PDF and query
    result = process_pdf_query(file, question)

    # Return the result
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
