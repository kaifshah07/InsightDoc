import os
import re
from PyPDF2 import PdfReader 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer


"""MOTO OF THE PROJECT - “From a large set of documents, find the most relevant information for a user question.”"""

#STEPS--upload → validate → save → extract text → verify output



# 1. Uploading document form the folder 

# loop for finding the pdf folder 
pdf_folder_path = "D:/Engennering/BE Project/InsightDoc_old/data/pdfs"
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
    full_text = "".join(char for char in full_text if char.isprintable())
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
Chunk 1 → 500 words  t
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
            "chunk_id": chunk_id,
            "source": doc_name,
            "text": chunk_text
        }
        all_chunks.append(chunk_data)
        
        start += (chunk_size - overlap)
        chunk_id += 1
print("The total chunks are :- ", len(all_chunks))
    
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
********* In our case the library sentence tranfroemr and the openai api doesnot worker , therefoer we shifted
to the other search and embedding techq called TF-IDF
"""
    
# # loading the model from secntence tranformer lib
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # loading the pre trained sementic model

# #to store the embedding chunks 
# embedded_chunks = []
# for chunk in all_chunks:  #LOOP THROUGH EACH CHUNK
#     chunk_text = chunk["text"]  #Extract the text from the chunk 
#     if not chunk_text:
#         continue
#     embedding_vector = embedding_model.encode(chunk_text) 
#     embedding_chunk = {  # cerating new dictonary 
#         "text" : chunk_text,
#         "embedding" : embedding_vector,
#         "source" : chunk["source"],
#         "chunk_id" : chunk["chunk_id"]
#     }
#     embedded_chunks.append(embedded_chunks)
# print("Total embedded chunks:- " , len(embedded_chunks))
# print("Embedding size: ", len(embedded_chunks[0]["embedding"]))

"""
1️⃣ INTUITION: WHAT TF-IDF REALLY DOES
Imagine you have 10 document chunks.
Some words appear:
everywhere → “the”, “is”, “and”
only in some documents → “initialization”, “authentication”
TF-IDF gives:
low importance to common words
high importance to rare but meaningful words
So the system learns:
“Rare words describe a document better than common words”
2️⃣ FORMAL MEANING (SIMPLE VERSION)
TF-IDF = Term Frequency × Inverse Document Frequency
🔹 Term Frequency (TF)
How often does a word appear in a document?
Example:
"system" appears 5 times in chunk A
→ higher TF
🔹 Inverse Document Frequency (IDF)
How rare is this word across all documents?
Example:
"system" appears in 9 out of 10 chunks → low IDF
"initialization" appears in 1 chunk → high IDF
🔹 Final Weight
TF-IDF = TF × IDF
Words that are:
✔ frequent in a chunk
✔ rare overall
→ get high scores
3️⃣ WHAT DOES TF-IDF PRODUCE?
It converts text → vector (numbers).
Example:
Chunk: "system initialization process"
TF-IDF vector:
[0.0, 0.34, 0.91, 0.12, ...]

Each number corresponds to a word.
📌 This is similar to embeddings, but:
statistical
not semantic
4️⃣ HOW SEARCH WORKS WITH TF-IDF
Step-by-step:
Convert all chunks → TF-IDF vectors
Convert user query → TF-IDF vector
Compare query vector with chunk vectors
Pick most similar chunks
Comparison is done using cosine similarity.
5️⃣ WHAT IS COSINE SIMILARITY? (INTUITIVE)
It measures:
“How similar is the direction of two vectors?”
Same direction → very similar
Different direction → not similar
Value range:
0 → not similar
1 → identical
6️⃣ HOW THIS FITS INTO YOUR PROJECT
Your updated pipeline becomes:
PDFs
 → Extract text
 → Clean text
 → Chunk text
 → TF-IDF vectors
 → Similarity search
 → Retrieve best chunks
📌 Exactly the same architecture as vector DBs.
7️⃣ WHAT WE WILL IMPLEMENT (CLEAR PLAN)
We will implement TF-IDF in 3 small steps:
STEP 1️⃣
Prepare chunk list
[
  {chunk_text, source_pdf, chunk_id},
  ...
]
STEP 2️⃣
Use TfidfVectorizer to:
Fit on all chunks
Generate vectors
STEP 3️⃣
Use cosine similarity to:
Compare query vs chunks
Return top-K chunks
"""

# now implementing the TF-IDF

"""But before code, understand what we need:
TF-IDF needs:
A list of texts
A vectorizer
A matrix of vectors"""

#1.Extact only text from the all_chunks
chunk_texts = [chunk["text"] for chunk in all_chunks]
print("The total chunks for TF-IDF are: " , len(chunk_texts))

#2. Create TF-IDF vectorizer 
TfidfVectorizer = TfidfVectorizer(
    max_features = 5000 ,     #limit vocabulary size
    stop_words="english"      #remove common words like the,this, and. 
)

# 3. Fit and tranform chunks into vectores 
tfidf_martix = TfidfVectorizer.fit_transform(chunk_texts)  #this is the main numerical rep of chunks 

"""
This is the explanation fo the martix below - tfidf_matrix = tfidf_vectorizer.fit_transform(chunk_texts)
This does two things:
Learns vocabulary from chunks
Converts each chunk into a vector
📌 Result:
Rows = chunks
Columns = words
Values = TF-IDF score
"""

#4. Verify the vectorization 
print("TF-IDF Matrix shape :-" ,tfidf_martix.shape)
# print("The chunk after the vectorization is - " , tfidf_martix)


"""
4️⃣ Shape verification
print(tfidf_matrix.shape)
Example output:
(120, 4321)
Meaning:
120 chunks
4321 unique meaningful words
✅ This confirms TF-IDF is working.
"""
"""Now we want to find the similarity between the words , by using the cosine similarty
Now we want:“Given a query, find the most relevant chunks”

WHAT NEW CONCEPT ARE WE ADDING?
🔹 Cosine Similarity
Measures angle between vectors
Output range: 0 → 1
Higher = more relevant
TF-IDF + cosine similarity = classic IR system

steps - Step 1: Take a user query
        Step 2: Convert query → TF-IDF vector
        Step 3: Compute similarity
        Step 4: Rank chunks by relevance
        Step 5: Fetch chunks using indices

"""
# Now moving on to the next step , we are goin to create Intreaction layer that will repeadty interact with t
# the user by using   a loop

top_k = 5
min_similarity = 0.1 #to filter the chunks below it , so they can be removed
# 1. User query
while True: # this lopp create the continous chating with the user
    query = input("\nEnter your Query. Type 'exit' to quit: " ).strip()
    if query.lower() in {"exit" , "quit"}:
        print("Exiting the search")
        break
    if not query:
        print("please enter avalid query!")
        continue

    #2. convert the query in to the tdf vector(same as vectorizer)
    query_vector = TfidfVectorizer.transform([query])

    #3. calculate cosine similiraty  between qurey and al chunks
    similarity_scores = cosine_similarity(query_vector,tfidf_martix)[0]
    
    #Filter chunks by minimum similarity 
    filtred_indices = [i for i , score in enumerate(similarity_scores) if score>= min_similarity]
    if not filtred_indices:
        print("No relevent chunks found for your Query..... \n Try again!!")
        continue
    # sort the filtered results by score descending
    filtred_indices.sort(key=lambda i:similarity_scores[i], reverse= True) 
        

    # #4. Get the similiraty score as a one D array
    # scores = similarity_scores[0]

    # #5. Get top 5 most revelant chunk 

    # top_indices = scores.argsort()[::-1][:top_k]

    #6. display the result 
    print("\nTop relevent chunks: \n")
    for rank ,idx in enumerate(filtred_indices[:top_k],start =1):
        chunk = all_chunks[idx]
        print(f"\nRank #{rank}")
        print(f"\nPDF Source: {chunk['source']}")
        print(f"Chunk ID:- {chunk['chunk_id']}")
        print(f"\nSimiliraty Scores:- {similarity_scores[idx]:.3f}")
        print("Text Preview:- ",chunk["text"][:500], "...")
        print("-" *80 )
        
        
        #old code before the ranking and cleaning the chunks 
        # print("Source_pdf:", all_chunks[idx]["source"])
        # print("chunk id :" , all_chunks[idx]["chunk_id"])
        # print("similiraty Scores:- ", score[idx])
        # print("Text:\n" , all_chunks[idx]["text"][:500]) #shows the frist 500 char
        # print("-" * 80)

"""The most iIMPORTATN line for the above code block to understand -TF-IDF retrieval works 
by converting both documents and user queries into numerical vectors, computing similarity 
scores between them, ranking the scores, and using the resulting indices to retrieve the most
relevant text chunks from the original dataset."""

"""
VERY IMPORTANT: HOW TO INTERPRET TF-IDF SCORES
TF-IDF score meanings (rule of thumb):
Score	Meaning
> 0.4	Strong keyword match
0.2 - 0.4	Relevant
0.1 - 0.2	Weak relevance
< 0.1	Mostly noise
"""


# Now moving on to the next step , we are goin to create Intreaction layer that will repeadty interact with t
# the user by using   a loop

""" now we move to -Result Filtering & Formatting 
-this is also done in the above code like min_similarity = 0.1 to fliter the most important chunks 
- 

Goal of this step
1. Filter out weak matches
Cosine similarity always returns a value, even for irrelevant chunks.
Introduce a minimum similarity threshold to discard chunks that are not meaningful.
2. Improve readability
Show fewer characters per chunk or add ellipsis (…)
Add clear ranking or numbering
Optionally, group by PDF
3. Make results trustable
Users only see results above a confidence level

Concepts
1️⃣ Minimum Similarity Threshold
Cosine similarity ranges from 0 (totally different) to 1 (identical).
A threshold (e.g., 0.1) removes chunks that are very different from the query.
This prevents showing irrelevant text.
2️⃣ Ranking & Presentation
Rank chunks by similarity (highest → lowest)
Show:
Rank (1,2,3…)
PDF source
Chunk ID
Similarity Score
Chunk preview (first 300-500 chars)
3️⃣ Optional Enhancements
Group results by PDF (so the user can see which documents contain the most relevant info)
Highlight query terms in the chunk (advanced, optional)

"""

