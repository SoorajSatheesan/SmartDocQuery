import os
import re
import streamlit as st
from ingestion import ingestion
from llm import get_response_from_query


def preprocess_response(response):
    """Cleans up the response text by removing metadata and formatting it."""
    if isinstance(response, str):
        # Remove metadata by splitting on "additional_kwargs" or similar patterns
        response = re.split(r"\nadditional_kwargs=|response_metadata=", response)[0]
        # Remove extra newlines and ensure clean formatting
        response = re.sub(r"\n\s*\n", "\n", response)  # Remove consecutive newlines
        response = re.sub(r"\n", " ", response)  # Replace remaining \n with spaces
        response = re.sub(r"\s+", " ", response).strip()  # Normalize spaces

        # Add formatting for better readability
        response = re.sub(
            r"\*\s*([^\*]+):", r"\n- **\1**:", response
        )  # Format bullet points
        response = re.sub(
            r"\s*-\s*", "\n- ", response
        )  # Ensure bullet points start on new lines

        return response.strip()
    return response


def ensure_directory_exists(directory):
    """Ensures the specified directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def user_interaction():
    api_key_file = "api_key.txt"
    retriever = None

    # Sidebar for settings and ingestion status
    st.sidebar.title("Settings and Status")

    # Step 1: Check if API key exists, otherwise prompt for one
    if not os.path.exists(api_key_file):
        st.sidebar.warning("API key not found. Please enter a valid API key below.")
        api_key = st.sidebar.text_input("Enter API key:", type="password")
        if st.sidebar.button("Save API key"):
            if api_key.strip():
                with open(api_key_file, "w") as f:
                    f.write(api_key.strip())
                st.sidebar.success("API key saved successfully.")
            else:
                st.sidebar.error("API key cannot be empty.")
    else:
        with open(api_key_file, "r") as f:
            api_key = f.read().strip()
        st.sidebar.success("API key loaded successfully.")

    # Step 2: File ingestion section in the sidebar
    st.sidebar.subheader("Ingest Document")
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["txt", "pdf"])

    if uploaded_file is not None:
        temp_dir = "temp"
        ensure_directory_exists(temp_dir)  # Ensure temp directory exists

        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"File uploaded: {file_path}")

        # Call the ingestion function
        retriever = ingestion(file_path)
        st.sidebar.success("File ingested successfully.")
    else:
        st.sidebar.info("No file selected.")

    # Main screen for chat-like interaction
    st.title("Chat Interface")

    st.write("Ask your questions below:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Enter your query:")
    if st.button("Submit Query"):
        if query.strip():
            with st.spinner("Processing your query..."):
                # Always send the query to the get_response_from_query function
                response = get_response_from_query(query, retriever)

            # Preprocess response to clean up metadata and format text
            clean_response = preprocess_response(response)

            # Append the query and clean response to the chat history (newest at the top)
            st.session_state.chat_history.insert(0, (query, clean_response))
        else:
            st.error("Query cannot be empty.")

    # Display chat history (most recent on top)
    for i, (q, r) in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {r}")


if __name__ == "__main__":
    user_interaction()
