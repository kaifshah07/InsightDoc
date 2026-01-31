import os
from PyPDF2 import PdfReader 
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Creating Chunks 
all_chunks = []
chunk_size = 500
overlap = 50

for doc_name, text in document_text.items():

    start = 0
    chunk_id = 0
    text_length = len(text)

    while start < text_length:

        chunk_text = text[start : start + chunk_size]
        #metadata
        chunk_data = {
            "chunk_id": chunk_id,
            "source": doc_name,
            "text": chunk_text
        }

        all_chunks.append(chunk_data)

        start += (chunk_size - overlap)
        chunk_id += 1
print("Chunking Starts!!")
print("The total chunks are:- " , len(all_chunks))
# print(all_chunks[36], all_chunks[2])

# We had tested the library for the embeddings which is sentence tranformers and the openai api , 
# But both did not prove right and then we shifted to TF-IDF which is give below (Main MODEL)

chunk_texts = [chunk["text"] for chunk in all_chunks]
print("The total chunks for TD-IDF are:-" , len(chunk_texts))

TfidfVectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)
tfidf_matrix = TfidfVectorizer.fit_transform(chunk_texts)
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
    query_vector = TfidfVectorizer.transform([query])
    
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
        
        