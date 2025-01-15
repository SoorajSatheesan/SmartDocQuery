# Agentic RAG Implementation using Gemini-1.5 FLASH

## Approach:

#### The implementation describes a very efficient but simple Agentic RAG pipeline leveraging open-source technologies combine with Gemini 1.5 Flash for efficient information retrieval. The process begins with document ingestion, supporting formats like PDF or TX. Documents are first preprocessed to convert them into text format if required and then split into chunks of size 1000 tokens with an overlap of 200 tokens. This overlapping allows continuity between chunks.

#### Then each chunk is converted into a vector representation using an open-source Sentence Transformer in this case “all-MiniLM-L6-v2”. These vector embeddings are stored in ChromaDB. When a user submits a query, the system first checks if the answer to the query is present in memory. If it is not, the query is embedded into the same vector space using the Sentence Transformer. A similarity search is then performed on ChromaDB using cosine similarity or other distance metrics to retrieve the most relevant document chunks. If no relevant document found, the query is sent to the default LLM, which generates a response, ensuring that even without uploaded documents, the system can provide a general but satisfactory reply to the user. To make the system user-friendly, a basic application was built using Streamlit.

## Design:

![](Design.png)

## Steps to setup:

### Step 1: Install required libraries:

`pip install langchain_community langchain pdfplumber sentence-transformers streamlit`

### Step 2: Enter your gemini API Key in api_key.txt

### Step 3: Navigate to the project folder and run :

`streamlit run main.py`
