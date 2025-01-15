from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain.schema import Document


# Embedding Model Class
class SentenceTransfromerEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return [
            embedding.tolist()
            for embedding in self.model.encode(texts, convert_to_tensor=True)
        ]

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=True).tolist()


def store_embeddings_in_chroma(
    documents, embedding_model, persist_directory="database"
):
    # Create chroma database
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
    print(f"Vector Database persisted")
    return vectordb


def load_vector_db(embedding_model, persist_directory="database"):
    return Chroma(
        persist_directory=persist_directory, embedding_function=embedding_model
    )
