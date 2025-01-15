import os
import pdfplumber
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def convert_pdf_to_txt(file_path: str) -> str:

    # Define the output directory
    output_dir = os.path.join(os.getcwd(), "documents")
    os.makedirs(
        output_dir, exist_ok=True
    )  # Create the 'documents' directory if it doesn't exist

    # Get file extension and base name
    file_extension = os.path.splitext(file_path)[1].lower()
    base_name = os.path.basename(file_path)

    if file_extension == ".txt":
        # Save the TXT file directly in the 'documents' directory
        dest_path = os.path.join(output_dir, base_name)
        with open(file_path, "r", encoding="utf-8") as src_file:
            content = src_file.read()
        with open(dest_path, "w", encoding="utf-8") as dest_file:
            dest_file.write(content)
        print(f"TXT file saved directly to: {dest_path}")
        return dest_path
    elif file_extension == ".pdf":
        # Convert PDF to TXT and save in the 'documents' directory
        txt_file_path = os.path.join(
            output_dir, f"{os.path.splitext(base_name)[0]}.txt"
        )

        with pdfplumber.open(file_path) as pdf:
            with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:  # Only write text if it's not None
                        txt_file.write(text + "\n")

        print(f"PDF successfully converted to TXT and saved to: {txt_file_path}")
        return txt_file_path
    else:
        # Raise an error for unsupported formats
        raise ValueError("Unsupported file format. Please provide a PDF or TXT file.")


def load_and_split_single_file(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:

    try:
        # Load a single text file
        loader = TextLoader(file_path, encoding="UTF-8")
        documents = loader.load()

        if not isinstance(documents, list):
            raise ValueError("Expected a list of documents from TextLoader.")

        print(f"Loaded {len(documents)} document(s) from {file_path}.")

        # Check if documents have the expected structure
        for doc in documents:
            if not isinstance(doc, Document):
                raise ValueError("Documents must be of type langchain.schema.Document.")

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Split the document(s) into chunks
        text_chunks = text_splitter.split_documents(documents)

        print(f"Split document into {len(text_chunks)} chunks.")
        return [chunk.page_content for chunk in text_chunks]
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
