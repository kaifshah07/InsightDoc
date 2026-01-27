import os
from PyPDF2 import PdfReader 
import re 
from sentence_transformers import SentenceTransformer

# document ingestion, batch processing, and text normalization pipeline 
pdf_folder_path = "data/pdfs"
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
print(all_chunks[0:5])

# loading the model from secntence tranformer lib
embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # loading the pre trained sementic model

#to store the embedding chunks (vectors of the chunk)
embedded_chunks = []
for chunk in all_chunks:  #LOOP THROUGH EACH CHUNK
    chunk_text = chunk["text"]  #Extract the text from the chunk 
    if not chunk_text:
        continue
    embedding_vector = embedding_model.encode(chunk_text) 
    embedding_chunk = {
        "text" : chunk_text,
        "embedding" : embedding_vector,
        "source" : chunk["source"],
        "chunk_id" : chunk["chunk_id"]
    }
    embedded_chunks.append(embedding_chunks)
    
print("the embedded vectores :" , embedded_chunks[0:2])
print("Total embedded chunks:- " , len(embedded_chunks))
print("Embedding size: ", len(embedded_chunks[0]["embedding"]))

# vector DB - Connection


    

