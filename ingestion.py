from preprocess import convert_pdf_to_txt
from preprocess import load_and_split_single_file
from embedding import store_embeddings_in_chroma
from embedding import load_vector_db
from embedding import SentenceTransfromerEmbeddings
from langchain.schema import Document

embedding_model = SentenceTransfromerEmbeddings()


def ingestion(path):
    documents_path = convert_pdf_to_txt(path)
    text = load_and_split_single_file(documents_path)
    documents = [Document(page_content=chunk) for chunk in text]
    vectordb = store_embeddings_in_chroma(documents, embedding_model)
    vectordb = load_vector_db(embedding_model)

    retriever = vectordb.as_retriever()

    print("Processed and Stored")
    return retriever