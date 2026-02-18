import os
import re
import pickle
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

INDEX_DIR = "index_data"
VECTORIZER_PATH = os.path.join(INDEX_DIR, "tfidf_vectorizer.pk1")
MATRIX_PATH = os.path.join(INDEX_DIR, "tfidf_matrix.npz")
CHUNK_PATH = os.path.join(INDEX_DIR, "chunks.pk1")

def save_index(vectorizer, tfidf_matrix, all_chunks):
    os.makedirs(INDEX_DIR, exist_ok=True)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    save_npz(MATRIX_PATH, tfidf_matrix)

    with open(CHUNK_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print("Index saved successfully.")

def sentence_based_chunking(document_text, max_chunk_size=500):

    all_chunks = []
    global_chunk_id = 0

    for doc_name, text in document_text.items():

        sentences = sent_tokenize(text)
        current_chunk = ""

        for sentence in sentences:

            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += " " + sentence
            else:
                all_chunks.append({
                    "chunk_id": global_chunk_id,
                    "source": doc_name,
                    "text": current_chunk.strip()
                })

                global_chunk_id += 1
                current_chunk = sentence

        if current_chunk.strip():
            all_chunks.append({
                "chunk_id": global_chunk_id,
                "source": doc_name,
                "text": current_chunk.strip()
            })

            global_chunk_id += 1

    return all_chunks

def build_index(pdf_folder_path):

    if not os.path.exists(pdf_folder_path):
        print("PDF folder not found.")
        return

    document_text = {}

    print("Scanning PDFs...")

    for file in os.listdir(pdf_folder_path):
        if file.lower().endswith(".pdf"):

            pdf_path = os.path.join(pdf_folder_path, file)
            reader = PdfReader(pdf_path)

            full_text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text

            full_text = re.sub(r"\n+", "\n", full_text)
            full_text = re.sub(r"\s+", " ", full_text)
            full_text = "".join(char for char in full_text if char.isprintable())
            full_text = full_text.strip()

            document_text[file] = full_text

    print("Total documents:", len(document_text))

    print("Creating chunks...")
    all_chunks = sentence_based_chunking(document_text)

    print("Total chunks:", len(all_chunks))

    chunk_texts = [chunk["text"] for chunk in all_chunks]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(chunk_texts)

    save_index(vectorizer, tfidf_matrix, all_chunks)

    print("Index building complete.")


if __name__ == "__main__":
    pdf_path = "D:/Engennering/BE Project/InsightDoc_old/data/pdfs"   # adjust to your project structure
    build_index(pdf_path)
    