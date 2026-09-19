import os
import re
import numpy as np

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pypdf import PdfReader

from google import genai
from google.genai import types


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in .env file")


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(api_key=GEMINI_API_KEY)

CHAT_MODEL = os.getenv(
    "GEMINI_CHAT_MODEL",
    "gemini-3.1-flash-lite"
)

EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001"
)


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# GLOBAL RAG STORAGE
# =========================================================

document_chunks = []
chunk_embeddings = []

uploaded_filename = None


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# SPLIT TEXT INTO CHUNKS
# =========================================================

def create_chunks(text, chunk_size=1200, overlap=200):

    text = clean_text(text)

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap

        if start < 0:
            start = 0

        if end >= len(text):
            break

    return chunks


# =========================================================
# CREATE EMBEDDING
# =========================================================

def create_embedding(text):

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    return np.array(
        response.embeddings[0].values,
        dtype=np.float32
    )


# =========================================================
# NORMALIZE VECTOR
# =========================================================

def normalize_vector(vector):

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


# =========================================================
# PROCESS PDF
# =========================================================

def process_pdf(file_path):

    global document_chunks
    global chunk_embeddings

    reader = PdfReader(file_path)

    full_text = ""

    for page_number, page in enumerate(reader.pages):

        try:

            page_text = page.extract_text()

            if page_text:
                full_text += "\n" + page_text

        except Exception as error:

            print(
                f"Error reading page {page_number + 1}: {error}"
            )

    full_text = clean_text(full_text)

    if not full_text:
        raise ValueError(
            "No readable text found in this PDF."
        )

    # Create chunks
    document_chunks = create_chunks(full_text)

    if not document_chunks:
        raise ValueError(
            "Could not create document chunks."
        )

    # Create embeddings
    chunk_embeddings = []

    for index, chunk in enumerate(document_chunks):

        print(
            f"Creating embedding {index + 1}/{len(document_chunks)}"
        )

        embedding = create_embedding(chunk)

        embedding = normalize_vector(embedding)

        chunk_embeddings.append(embedding)

    print(
        f"PDF processed successfully. "
        f"{len(document_chunks)} chunks created."
    )


# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(vector_a, vector_b):

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0

    return float(
        np.dot(vector_a, vector_b) / denominator
    )


# =========================================================
# SEARCH PDF
# =========================================================

def search_pdf(question, top_k=5):

    if not document_chunks:
        return []

    question_embedding = create_embedding(question)

    question_embedding = normalize_vector(
        question_embedding
    )

    scores = []

    for index, chunk_embedding in enumerate(
        chunk_embeddings
    ):

        score = cosine_similarity(
            question_embedding,
            chunk_embedding
        )

        scores.append(
            (score, index)
        )

    scores.sort(
        key=lambda x: x[0],
        reverse=True
    )

    results = []

    for score, index in scores[:top_k]:

        results.append(
            {
                "score": score,
                "text": document_chunks[index]
            }
        )

    return results


# =========================================================
# CHECK IF PDF HAS RELEVANT INFORMATION
# =========================================================

def is_relevant(results, threshold=0.42):

    if not results:
        return False

    best_score = results[0]["score"]

    print(
        f"Best similarity score: {best_score}"
    )

    return best_score >= threshold


# =========================================================
# GENERATE ANSWER USING ONLY PDF CONTEXT
# =========================================================

def generate_answer(question, results):

    context_parts = []

    for result in results:

        context_parts.append(
            result["text"]
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a strict PDF Question Answering assistant.

IMPORTANT RULES:

1. Answer ONLY using the information provided in the PDF context below.
2. Do NOT use your own general knowledge.
3. Do NOT use information from the internet.
4. Do NOT guess.
5. Do NOT make up information.
6. If the answer is not clearly available in the PDF context,
   reply exactly:

Sorry, I couldn't find the answer in the uploaded PDF.

7. Keep the answer directly related to the user's question.
8. You can summarize or explain information from the PDF,
   but do not introduce outside information.
9. If the user asks a general question that is unrelated to
   the uploaded PDF, do not answer it.
10. The uploaded PDF is the ONLY source of truth.

PDF CONTEXT:
-------------------------
{context}
-------------------------

USER QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0
        )
    )

    return response.text.strip()


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# UPLOAD PDF
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_pdf():

    global uploaded_filename

    if "document" not in request.files:

        return jsonify(
            {
                "success": False,
                "message": "Please select a PDF file."
            }
        )

    file = request.files["document"]

    if file.filename == "":

        return jsonify(
            {
                "success": False,
                "message": "Please select a PDF file."
            }
        )

    if not file.filename.lower().endswith(".pdf"):

        return jsonify(
            {
                "success": False,
                "message": "Only PDF files are allowed."
            }
        )

    # Save PDF
    filename = file.filename

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(file_path)

    try:

        process_pdf(file_path)

        uploaded_filename = filename

        return jsonify(
            {
                "success": True,
                "message": (
                    f"{filename} uploaded successfully. "
                    f"You can now ask questions from this PDF."
                ),
                "chunks": len(document_chunks)
            }
        )

    except Exception as error:

        print(
            f"PDF processing error: {error}"
        )

        return jsonify(
            {
                "success": False,
                "message": (
                    "Could not process the PDF. "
                    "Please upload a readable PDF."
                )
            }
        )


# =========================================================
# CHAT
# =========================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    if not document_chunks:

        return jsonify(
            {
                "answer": (
                    "Please upload a PDF first."
                )
            }
        )

    data = request.get_json()

    question = data.get(
        "message",
        ""
    ).strip()

    if not question:

        return jsonify(
            {
                "answer": "Please enter a question."
            }
        )

    try:

        # Search relevant PDF chunks
        results = search_pdf(
            question,
            top_k=5
        )

        # Check whether question is actually related
        # to uploaded PDF
        if not is_relevant(
            results,
            threshold=0.42
        ):

            return jsonify(
                {
                    "answer": (
                        "Sorry, I couldn't find the answer "
                        "in the uploaded PDF."
                    )
                }
            )

        # Generate answer only from retrieved PDF content
        answer = generate_answer(
            question,
            results
        )

        return jsonify(
            {
                "answer": answer
            }
        )

    except Exception as error:

        print(
            f"Chat error: {error}"
        )

        return jsonify(
            {
                "answer": (
                    "Sorry, an error occurred while "
                    "processing your question."
                )
            }
        )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=True
    )