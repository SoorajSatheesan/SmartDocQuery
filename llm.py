from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import json


class MemoryManager:
    def __init__(self, max_memory_size=10):
        self.max_memory_size = max_memory_size
        self.memory = []  # Store memory as a list of dictionaries

    def add_to_memory(self, query, response):
        """Adds a query and response to memory, ensuring it doesn't exceed max size."""
        # Ensure response is serializable before adding
        self.memory.append({"query": query, "response": str(response)})
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)  # Remove the oldest memory
        print("Response stored in memory.")

    def get_from_memory(self, query):
        """Checks if a query exists in memory and returns the response if found."""
        for item in self.memory:
            if query.lower() in item["query"].lower():
                return item["response"]
        return None

    def save_memory_to_file(self, file_path="memory.json"):
        """Saves the memory to a JSON file."""
        with open(file_path, "w") as file:
            json.dump(self.memory, file)

    def load_memory_from_file(self, file_path="memory.json"):
        """Loads memory from a JSON file."""
        try:
            with open(file_path, "r") as file:
                self.memory = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            self.memory = []


def read_api_key(file_path="api_key.txt"):
    """Reads the API key from a file."""
    with open(file_path, "r") as file:
        api_key = file.read().strip()
    return api_key


def initialize_qa_chain(
    retriever=None,
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
):
    """Initializes the RetrievalQA chain and LLM."""
    # Read API key
    api_key = read_api_key()

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )

    # Create the QA chain only if retriever is provided
    if retriever:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        return qa_chain, llm

    return None, llm


def chain_of_thought_reasoning(query, retriever, memory_manager, llm):
    """Performs reasoning to determine retrieval vs memory use."""
    # Check memory for the response first
    response = memory_manager.get_from_memory(query)
    if response:
        return f"{response}"

    # Attempt retrieval if retriever is provided and memory is insufficient
    if retriever:
        try:
            qa_chain = initialize_qa_chain(retriever)[0]
            llm_response = qa_chain.invoke({"query": query})

            if llm_response.get("result"):
                memory_manager.add_to_memory(query, llm_response["result"])
                return f"{llm_response['result']}"
        except Exception as e:
            print(f"Retrieval error: {e}")

    # Fallback to direct LLM response if memory and retrieval are insufficient
    try:
        llm_response = llm.invoke(query)  # Pass the query directly as a string
        return f"{llm_response}"
    except Exception as e:
        return f"An error occurred while processing your query: {e}"


def get_response_from_query(query, retriever=None):
    """Gets a response to a query by using reasoning and retrieval."""
    # Initialize memory manager and LLM
    memory_manager = MemoryManager()
    memory_manager.load_memory_from_file()
    _, llm = initialize_qa_chain(retriever)

    # Perform reasoning and retrieval
    response = chain_of_thought_reasoning(query, retriever, memory_manager, llm)

    # Save the updated memory to file
    if "Response from retrieval:" in response:
        memory_manager.save_memory_to_file()

    # Output the response
    return response
