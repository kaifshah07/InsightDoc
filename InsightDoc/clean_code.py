import os
from PyPDF2 import PdfReader 
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama
import pickle
from scipy.sparse import save_npz, load_npz
import nltk 
nltk.download("punkt")
from nltk.tokenize import sent_tokenize


INDEX_DIR = "index_data"
VECTORIZER_PATH = os.path.join(INDEX_DIR, "tfidf_vectorizer.pk1")
MATRIX_PATH =   os.path.join(INDEX_DIR ,"tfidf_matrix.npz")
CHUNK_PATH = os.path.join(INDEX_DIR,"chunks.pk1")

def index_exists():
    return(
        os.path.exists(VECTORIZER_PATH)
        and os.path.exists(MATRIX_PATH)
        and os.path.exists(CHUNK_PATH)
    )

def save_index(vectorizer,tfidf_matrix,all_chunks):
    os.makedirs(INDEX_DIR,exist_ok=True)
    
    with open(VECTORIZER_PATH,"wb") as f:
        pickle.dump(vectorizer,f)
    save_npz(MATRIX_PATH,tfidf_matrix)
    with open(CHUNK_PATH,"wb") as f:
        pickle.dump(all_chunks,f)
        
    print("Index saved to disk successfully...")
    
def load_index():
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    tfidf_matrix = load_npz(MATRIX_PATH)

    with open(CHUNK_PATH, "rb") as f:
        all_chunks = pickle.load(f)

    print("Index loaded from disk successfully.")
    return vectorizer, tfidf_matrix, all_chunks


if index_exists():
    print("Existing index found. Loading........")
    vectorizer,tfidf_matrix,all_chunks = load_index()
else:
    print("NoIndex Found From Pdfs.......")
# document ingestion, batch processing, and text normalization pipeline 
    pdf_folder_path = "D:/Engennering/BE Project/InsightDoc_old/data/pdfs"
    print("The current working directory" , os.getcwd())
    if os.path.exists(pdf_folder_path):
        print("Pdf folder found ")
    else:
        print("Pdf folder not found ")
        exit()
    files  = os.listdir(pdf_folder_path)  


    pdf_files = [] 
    for file in files:
        if file.lower().endswith(".pdf"):
            pdf_files.append(file) 


    document_text = {}
    print("Starting the pdf scan one by one for all PDFS")
    for pdf_file in pdf_files:
        print("\n Processing...........", pdf_file) 

        pdf_path = os.path.join(pdf_folder_path,pdf_file)  
        reader = PdfReader(pdf_path)  
        full_text = ""   

        for page in reader.pages: 
            page_text = page.extract_text()
            if page_text:
                full_text += page_text

        full_text = re.sub(r"\n+" , "\n" , full_text)
        full_text = re.sub(r"\s+", " " , full_text)
        full_text = "".join(char for char in full_text if char.isprintable())
        full_text = full_text.strip()

        document_text[pdf_file] = full_text  #stores the extracted text of each PDF using its filename as an identifier.

    print("\n Processing summary..............")
    print("Total document processed: ", len(document_text)) 

    for doc_name, text in document_text.items(): 
        print(doc_name," -> text length ", len(text))

    # Now senence based chunking 
    def sentence_based_chunking(document_text, max_chunk_size=500):
        all_chunks = []

        for doc_name, text in document_text.items():
            sentences = sent_tokenize(text)

            current_chunk= ""
            global_chunk_id = 0

            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= max_chunk_size:
                    current_chunk += " " + sentence
                else:
                    all_chunks.append({
                        "chunk_id" : chunk_id,
                        "source" : doc_name,
                        "text" : current_chunk.strip()
                    })
                    chunk_id +=1
                    current_chunk = sentence #start new chunk
            if current_chunk.strip():
                all_chunks.append({
                    "chunk_id" : chunk_id,
                    "source" : doc_name,
                    "text" : current_chunk.strip()
                })
        return all_chunks
    print("Creating sentence based chunks ....")
    all_chunks = sentence_based_chunking(document_text,max_chunk_size=500)
    print("Total chunks Created : " , len(all_chunks))
    # We had tested the library for the embeddings which is sentence tranformers and the openai api , 
    # But both did not prove right and then we shifted to TF-IDF which is give below (Main MODEL)

    chunk_texts = [chunk["text"] for chunk in all_chunks]
    print("The total chunks for TD-IDF are:-" , len(chunk_texts))
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(chunk_texts)
    save_index(vectorizer, tfidf_matrix, all_chunks)

print("TF-IDF martix shape:-" , tfidf_matrix.shape)
top_k = 5
min_similarity = 0.1
while True:
    query = input("\nEnter your Query. Type 'exit' to quit: " ).strip()
    if query.lower() in {"exit" , "quit"}:
        print("Exiting the search")
        break
    if not query:
        print("please enter avalid query!")
        continue
    query_vector = vectorizer.transform([query])

    
    similarity_scores = cosine_similarity(query_vector,tfidf_matrix)[0]
    filtred_indices = [i for i , score in enumerate(similarity_scores) if score>= min_similarity]
    if not filtred_indices:
        print("No relevent chunks found for your Query..... \n Try again!!")
        continue
    filtred_indices.sort(key=lambda i:similarity_scores[i], reverse= True)
    print("\nTop relevent chunks: \n")
    for rank ,idx in enumerate(filtred_indices[:top_k],start =1):
        chunk = all_chunks[idx]
        print(f"\nRank #{rank}")
        print(f"\nPDF Source: {chunk['source']}")
        print(f"Chunk ID:- {chunk['chunk_id']}")
        print(f"\nSimiliraty Scores:- {similarity_scores[idx]:.3f}")
        print("Text Preview:- ",chunk["text"][:500], "...")
        print("-" *80 )
        
    # CONTEXT ASSEMBLY
    print("Now Context Assembly start.")
    context_char_lim = 5000
    context = ""
    for idx in filtred_indices:
        if similarity_scores[idx] < min_similarity:
            break

        chunk = all_chunks[idx]
        chunk_text = (
            f"[Source: {chunk['source']}, Chunk: {chunk['chunk_id']}]\n"
            f"{chunk['text']}\n\n"
        )

        if len(context) + len(chunk_text) > context_char_lim:
            break

        context += chunk_text

    print("\nAssembeld Context for LLM :-")
    # print(context[:1000], "...\n") 
    """
    You are an information extraction assistant.
    Answer the user's question using ONLY the information present in the context below.
    DO NOT add new information.
    DO NOT rephrase the question.
    DO NOT invent examples.
    If the answer is not explicitly present in the context, say:
    "Answer not found in the provided documents."
"""
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
    print(answer)
    
    context_parts = []
    source_metadata = []
    
    for rank, idx in enumerate(filtred_indices[:top_k], start=1):
        chunk = all_chunks[idx]
    
        context_parts.append(
            f"[Source {rank} | {chunk['source']} | Chunk {chunk['chunk_id']}]\n"
            f"{chunk['text']}"
        )
    
        source_metadata.append({
            "rank": rank,
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"]
        })
    
    # context = "\n\n".join(context_parts)
    
    print("\nSources Used:\n")

    for source in source_metadata:
        print(
            f"- Source {source['rank']}: "
            f"{source['source']} (Chunk {source['chunk_id']})"
        )
    

    
    
    