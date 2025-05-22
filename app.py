import os
import sqlite3
import hashlib
import uuid
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
import nltk
from nltk.tokenize import sent_tokenize
import fitz  # PyMuPDF
from docx import Document
from PIL import Image
import pytesseract
import logging

# --- CONFIG ---
UPLOAD_ROOT = 'uploads'
VECTOR_DB_ROOT = 'vector_db'
DB_PATH = 'users.db'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg'}
SECRET_KEY = 'your_secret_key_here'  # Change this!

# --- INIT ---
app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app)
os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(VECTOR_DB_ROOT, exist_ok=True)
nltk.download('punkt')

# --- DB SETUP ---
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            upload_folder TEXT NOT NULL,
            vector_db_path TEXT NOT NULL
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            message TEXT,
            response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )''')
        conn.commit()
init_db()

# --- GEMINI & EMBEDDINGS ---
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    try:
        if ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == 'pdf':
            doc = fitz.open(file_path)
            text = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img)
                text.append(ocr_text)
            return '\n'.join(text)
        elif ext == 'docx':
            doc = Document(file_path)
            return '\n'.join([p.text for p in doc.paragraphs])
        elif ext in ['png', 'jpg', 'jpeg']:
            image = Image.open(file_path)
            return pytesseract.image_to_string(image)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        return ""

def get_user():
    if 'user_id' not in session:
        return None
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, upload_folder, vector_db_path FROM users WHERE id=?", (session['user_id'],))
        row = c.fetchone()
        if row:
            return {'id': row[0], 'username': row[1], 'upload_folder': row[2], 'vector_db_path': row[3]}
        return None

def get_chroma_collection(vector_db_path):
    chroma_client = chromadb.PersistentClient(path=vector_db_path)
    return chroma_client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )

# --- ROUTES ---

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        password_hash = generate_password_hash(password)
        user_uuid = str(uuid.uuid4())
        upload_folder = os.path.join(UPLOAD_ROOT, user_uuid)
        vector_db_path = os.path.join(VECTOR_DB_ROOT, user_uuid)
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(vector_db_path, exist_ok=True)
        try:
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (username, password_hash, upload_folder, vector_db_path) VALUES (?, ?, ?, ?)",
                          (username, password_hash, upload_folder, vector_db_path))
                conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('signup.html', error="Username already exists.")
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT id, password_hash FROM users WHERE username=?", (username,))
            row = c.fetchone()
            if row and check_password_hash(row[1], password):
                session['user_id'] = row[0]
                return redirect(url_for('index'))
        return render_template('login.html', error="Invalid credentials.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    user = get_user()
    if not user:
        return redirect(url_for('login'))
    return render_template('index.html', username=user['username'])

@app.route('/api/upload', methods=['POST'])
def upload_file():
    user = get_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    files = request.files.getlist('files')
    processed_files = []
    errors = []
    collection = get_chroma_collection(user['vector_db_path'])
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(user['upload_folder'], filename)
                file.save(filepath)
                text = extract_text(filepath)
                if not text:
                    raise ValueError("No text extracted")
                sentences = sent_tokenize(text)
                embeddings = embedding_model.encode(sentences)
                collection.add(
                    embeddings=embeddings.tolist(),
                    documents=sentences,
                    metadatas=[{"source": filename}] * len(sentences),
                    ids=[f"{filename}_{i}" for i in range(len(sentences))]
                )
                processed_files.append(filename)
            except Exception as e:
                errors.append(f"Error processing {file.filename}: {str(e)}")
    if errors:
        return jsonify({'error': '; '.join(errors)}), 500
    return jsonify({'message': 'Files uploaded and processed successfully', 'processed_files': processed_files})

@app.route('/api/chat', methods=['POST'])
def chat():
    user = get_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    collection = get_chroma_collection(user['vector_db_path'])
    query_embedding = embedding_model.encode(query)
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=3)
    context = "\n\n".join(results['documents'][0]) if results['documents'] else ""
    prompt = f"""You are a helpful AI assistant. Use the following context to answer the question. 
    Format your response in a clear, structured way using markdown:
    - Use bullet points for lists
    - Use headings for main sections
    - Use bold for emphasis
    - Keep paragraphs concise

    Context:
    {context}

    Question: {query}

    Answer:"""
    response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
    answer = response.text if hasattr(response, 'text') else str(response)
    # Store chat history
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO chat_history (user_id, message, response) VALUES (?, ?, ?)",
                  (user['id'], query, answer))
        conn.commit()
    return jsonify({'response': answer})

@app.route('/api/files', methods=['GET'])
def list_files():
    user = get_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    files = []
    for filename in os.listdir(user['upload_folder']):
        if allowed_file(filename):
            file_path = os.path.join(user['upload_folder'], filename)
            files.append({
                'name': filename,
                'size': os.path.getsize(file_path),
                'uploaded_at': os.path.getmtime(file_path),
                'type': filename.rsplit('.', 1)[1].lower()
            })
    files.sort(key=lambda x: x['uploaded_at'], reverse=True)
    return jsonify({'files': files})

@app.route('/api/files/<path:filename>', methods=['DELETE'])
def delete_file(filename):
    user = get_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    filename = secure_filename(filename)
    file_path = os.path.join(user['upload_folder'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    os.remove(file_path)
    collection = get_chroma_collection(user['vector_db_path'])
    collection.delete(where={"source": filename})
    return jsonify({'message': 'File deleted successfully'})

@app.route('/api/history', methods=['GET'])
def chat_history():
    user = get_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT message, response, timestamp FROM chat_history WHERE user_id=? ORDER BY timestamp DESC", (user['id'],))
        history = [{'message': row[0], 'response': row[1], 'timestamp': row[2]} for row in c.fetchall()]
    return jsonify({'history': history})

@app.route('/uploads/<path:filename>')
def serve_document(filename):
    user = get_user()
    if not user:
        return "Not authenticated", 401
    return send_from_directory(user['upload_folder'], filename)

@app.route('/logout')
def logout_user():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5001)