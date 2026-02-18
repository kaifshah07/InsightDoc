import os
import pickle
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import ollama


INDEX_DIR = "index_data"
VECTORIZER_PATH = os.path.join(INDEX_DIR, "tfidf_vectorizer.pk1")
MATRIX_PATH =   os.path.join(INDEX_DIR ,"tfidf_matrix.npz")
CHUNK_PATH = os.path.join(INDEX_DIR,"chunks.pk1")

def load_index():
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    tfidf_matrix = load_npz(MATRIX_PATH)

    with open(CHUNK_PATH, "rb") as f:
        all_chunks = pickle.load(f)

    print("Index loaded from disk successfully.")
    return vectorizer, tfidf_matrix, all_chunks
def retrieve_chunks(query, vectorizer, tfidf_matrix,
                    min_similarity=0.1, top_k=5):

    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]

    filtered_indices = [
        i for i, score in enumerate(similarity_scores)
        if score >= min_similarity
    ]

    filtered_indices.sort(
        key=lambda i: similarity_scores[i],
        reverse=True
    )

    return filtered_indices[:top_k], similarity_scores
def build_context(indices, all_chunks,
                  similarity_scores, max_chars=5000):

    context = ""

    for idx in indices:
        chunk = all_chunks[idx]

        chunk_text = (
            f"[Source: {chunk['source']}, "
            f"Chunk: {chunk['chunk_id']}]\n"
            f"{chunk['text']}\n\n"
        )

        if len(context) + len(chunk_text) > max_chars:
            break

        context += chunk_text

    return context


def generate_answers(query,context, model="qwen:1.8b"):
     # Prompt Construction
    prompt = f"""
    You are a document question-answering system.
         Your task is to find the answer in the provided document content.
         Rules (mandatory):
            - Use ONLY the information explicitly stated in the Context.
            - Do NOT add, expand, explain, or interpret.
            - Answer at the SAME LEVEL OF DETAIL as requested in the question.
            - When listing items, list ONLY what is explicitly asked.
            - If the question asks for names, return ONLY the names.
            - Do NOT include sub-points unless they are explicitly requested.
            - If the answer is not explicitly present, respond exactly:
              Not found in the document.
    Context:
    {context}

    User Question:
    {query}

    Answer (bullet points if applicable):  
    """
    print("The LLM is processing and wroking on an answer.........")
    # print(prompt[:1000] )
    
    response  = ollama.chat(
    model="qwen:1.8b",    
    messages=[
            {"role" : "user",
             "content" : prompt}
    ],
    options = {
    "temperature": 0.0,      # ❗ no creativity
    "top_p": 1.0,
    "top_k": 20,
    "num_predict": 120,      # ❗ short factual answers
    "repeat_penalty": 1.05
    }
        
    )
    answer = response["message"]["content"].strip()
    print("\nLLM Generated answer: ")
    # print(answer)
    return answer 