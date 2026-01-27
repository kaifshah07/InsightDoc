import os
import re
from PyPDF2 import PdfReader 
from sentence_transformers import SentenceTransformer


"""MOTO OF THE PROJECT - “From a large set of documents, find the most relevant information for a user question.”"""

#STEPS--upload → validate → save → extract text → verify output



# 1. Uploading document form the folder 

# loop for finding the pdf folder 
pdf_folder_path = "data/pdfs"
print("The current working directory" , os.getcwd())
if os.path.exists(pdf_folder_path):
    print("Pdf folder found ")
else:
    print("Pdf folder not found ")
files  = os.listdir(pdf_folder_path) # listing all the file in the dir 

#Loop for finding all the pdfs in the file and accessing them
pdf_files = [] # list used to store and filter all the pdfs form the folder above 
for file in files:
    if file.lower().endswith(".pdf"):
        pdf_files.append(file) 
print("Pdf from this folder are listed below")
# for pdf in pdf_files:
#     print(pdf)
    
"""
Below following things are done - with the help of Pdfreader lib 
1. PDF parsing
2. Page-level access
3. Text extraction
4. File path joining
5. Content verification
"""

# 2. Now the selection of the pdf and extraction of the text form one pdf at a time  

# frist_pdf = pdf_files[7] # picking one pdf for text extraction 
# pdf_path = os.path.join(pdf_folder_path,frist_pdf) # combine the pdf path and pdf from the folder
# reader = PdfReader(pdf_path) # to read the pdf by using the library 
# """
# To read the first page only of the pdf
# frist_page = reader.pages[0] # to read the frsit page of the pdf with the index 0 
# text = frist_page.extract_text() # now extracting the text form the fist page 
# print(text)
# """
# full_text = "" # for storing the txt form all the pages of the pdf and then printing with required num of char
# total_pages = len(reader.pages) # to find the total number of pages of the pdf 

# # for loop for extracting the text form all the pages of the pdf
# for page_num in range(total_pages):
#     page = reader.pages[page_num]
#     page_text = page.extract_text()
    
#     if page_text: #it prevents adding none if there is no text found on the page
#         full_text += page_text
# print("----------The Extracted Text From The Pdf with the name , `" ,frist_pdf,"`--------------------------" )
# print("Total pages in the pdf are: ",total_pages)
# print("Total extracted page lenght : ", len(full_text))
# print("\nPreview (first 300 character) ")
# # print(full_text[:1000])


""" Previously we have done text extraction for one pdf only, above is the commented  
code for one pdf text extraction and one pdf only  
But for the extraction of all the pdfs at the same time or to Process all pdf one by one and store the 
characters we use the following for lopp 
-we need to make a set for storing all the characters from the pdf in it and then make a for loop 
-for the pdfreader to read one pdf at a time and store it in var called full_text with all its pages char 
then at last we assign the one pdf extracted character as a list and then store this list in the 
SET called documet_text = {} 
 
 IMPORTANT - "document_text[pdf_file] = full_text" -> here the readed pdfs characters are stored in the full_text
             variable and then this full_text is stored in the main set and 
             assign as per the pdfs coming next to it , for every pdf 
             the full_text var is different and then this full_text varible is stored in the main set called 
             "document_text" where all the pdf extracted chr are stored in the index wise manner 
 """

document_text = {} # Here the SET is used to store all the pdf extracted text as it can store int & num data both 

print("Starting the pdf scan one by one for all PDFS")
for pdf_file in pdf_files:
    print("\n Processing...........", pdf_file) 
    
    pdf_path = os.path.join(pdf_folder_path,pdf_file) # path for the pdfs to read 
    reader = PdfReader(pdf_path) # here the pdf is selected to read 
    full_text = ""  # here the selected pdf char is stored , all char 
    
    for page in reader.pages: # for loop is used to read/extract the char from the pdfs
        page_text = page.extract_text()
        if page_text:
            full_text += page_text
    #1. To replace the new line with a single newline 
    full_text = re.sub(r"\n+" , "\n" , full_text)
    #2. Replace mulptiple spaces with a single space
    full_text = re.sub(r"\s+", " " , full_text)
    #3. Remove non-printable char 
    full_text = " ".join(char for char in full_text if char.isprintable())
    #4. Remove leading and traling spaces 
    full_text = full_text.strip()
    
    document_text[pdf_file] = full_text #document_text → { pdf_name : cleaned_full_text } this is structure of it.
    # here we assign all the char from one pdf as per the index in a list then 
    #store this list of one pdf characters in the set in the frist index and same of the coming 
    # next pdf * important 
    
print("\n Processing summary..............")
print("Total document processed: ", len(document_text)) # the result is printed after reading all the pfd char
for doc_name, text in document_text.items(): # here all the iteams in the main SET is printed in the series
    print(doc_name," -> text length ", len(text))
 
"""
“All PDFs are processed and text is stored.”
Till here all the complete local document ingestion & extraction system is complete
and Raw text extracted from PDFs Nd stored in the set as list 

The next step is “Take the extracted text and convert it into clean, consistent, AI-ready text.” 
CLEANING PIPELINE (HIGH-LEVEL)

We had appled cleaning in layers:
Normalize whitespace
Remove unnecessary newlines
Remove non-printable characters
Lowercase text (optional)
Keep text readable
⚠️ We will not remove punctuation yet.

Psudo code for it 
For each document text:
    Replace multiple newlines with single newline
    Replace multiple spaces with single space
    Remove non-printable characters
    Strip leading and trailing spaces
    Store cleaned text
    
now we can say that - 
“Implemented document ingestion, batch processing, and text normalization pipeline for NLP and RAG systems.”
That line alone is resume gold.
this is done in above code 
"""
"""
Now the next step is text chuncking , which is the process of breaking the long document text into small.meaning
full peices of text -
Instead of this - 10,000 words in one block
We do this-
Chunk 1 → 500 words  
Chunk 2 → 500 words  
Chunk 3 → 500 words  
Each chunk becomes an independent unit.
A chunk is:A small piece of text With context Of fixed size
Typical sizes:
300500 words
or 800-1000 characters
For beginners:
👉 Character-based chunking is simplest and safest or Chunking = turning a book into readable pages.
-Updated pipeline: 
 ->PDFs
 → Extract text
 → Clean text
 → CHUNK text ✅
 → Store chunks + metadata
---- After chunking, you unlock:
Embeddings
Vector databases
RAG
AI-powered search
Chat over documents
Without chunking → AI project fails

VECTOR DB - A Vector DB stores text in a way that allows semantic (meaning-based) search, not keyword search 
           that is the reason vector db is used for ai based search 
          - HOW DOES A VECTOR DB DO THIS ? By storing embeddings instead of raw text
EMBEDDINGS - WHAT IS AN EMBEDDING? Simple definition:
            "An embedding is a numerical representation of meaning."
           - for eg-Text:"The server starts automatically"
                    Embedding: [0.12, -0.45, 0.88, 0.03, ...]
                        Hundreds or thousands of numbers
                        Similar meanings → similar vectors
                        📌 You NEVER write embeddings manually.
MTEA DATA - 🔹 WHAT IS METADATA?
            Definition:Metadata = data about data
                                For each chunk, metadata stores:
                                Source document name
                                Chunk index
                                Page number (optional)
                                Section (optional)
    
    -----Now for the next step this is the structure for chuncking and doing all the meangin full work -------
                    [ Chunk Text ]
                          ↓
                    [ Embedding Model ]
                          ↓
                    [ Vector ]
                          ↓
                    [ Vector Database ]
                          ↓
                    Store vector + metadata
                    
-So the VECTOR DB  is the main back bone for the stroing and precessing all the text and it stores chunk text
, embeddings ,metadata 

🔍 HOW SEARCH WORKS (STEP-BY-STEP)
User asks:
“How does the system initialize?”
Steps:
Convert question → embedding
Compare question vector with all stored vectors
Find closest matches (cosine similarity)
Retrieve top chunks
Send chunks to LLM
✔ Accurate answer
✔ Source-based
✔ Fast
"""
#LIST TO STORE ALL CHUNKS and each chunk will be dict
all_chunks = []
chunk_size = 500
overlap = 50
for doc_name,text in document_text.items():
    start = 0
    chunk_id = 0
    text_length = len(text)
    while start < text_length:      # extracting one chunk saftely 
        chunk_text = text[start : start + chunk_size] # Slicing here for the text in the chunk for processing  
        chunk_data = {
            "chunk_id" : chunk_id
            "source" :doc_name,
            "text" : chunk_text,
        }
        all_chunks.append(chunk_data)
        
        start += (chunk_size - overlap)
        chunk_id += 1
    
# all this is done now 

"""
Now Embeddings Generation is goinn to be done inthe below code chunk 
# we use pre trained embeddings model , and not make them , it take input and gives vector of num 
#Examples (conceptually):
#OpenAI embeddings
#SentenceTransformers
#HuggingFace models
#📌 You don't train this model — you use a pre-trained one.

🔹 STEP 2: DECIDE INPUT UNIT (IMPORTANT)
What goes into the model?
👉 Only the chunk text

sementic search -
WHAT EMBEDDINGS ENABLE (VERY IMPORTANT) -Once embeddings exist, you can:
Convert a user query into an embedding
Compare it with stored embeddings
Find most similar chunks
Send them to an LLM
This is semantic search.

Chunk text
   ↓
Embedding model
   ↓
Vector (numbers)
   ↓
Store with metadata

--- we are using embeding library - sentence-transformers 
"""
    
# loading the model from secntence tranformer lib
embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # loading the pre trained sementic model

#to store the embedding chunks 
embedded_chunks = []
for chunk in all_chunks:  #LOOP THROUGH EACH CHUNK
    chunk_text = chunk["text"]  #Extract the text from the chunk 
    if not chunk_text:
        continue
    embedding_vector = embedding_model.encode(chunk_text) 
    embedding_chunk = {  # cerating new dictonary 
        "text" : chunk_text,
        "embedding" : embedding_vector,
        "source" : chunk["source"],
        "chunk_id" : chunk["chunk_id"]
    }
    embedded_chunks.append(embedded_chunks)
print("Total embedded chunks:- " , len(embedded_chunks))
print("Embedding size: ", len(embedded_chunks[0]["embedding"]))

