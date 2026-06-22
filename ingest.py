import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"          # where the vector DB will be persisted on disk
EMBED_MODEL = "nomic-embed-text"  # a small, fast local embedding model served by Ollama


def load_documents():
    """Load every .txt file inside the data/ folder."""
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from '{DATA_DIR}'")
    return documents


def split_documents(documents):
    """
    Break long documents into overlapping chunks.
    Overlap helps preserve context across chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s)")
    return chunks


def build_vector_store(chunks):
    """
    Embed each chunk and store it in ChromaDB.
    This persists to disk so we don't have to re-embed every time the app starts.
    """
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="knowledge_base",
    )
    print(f"Vector store created and saved at '{CHROMA_DIR}'")
    return vector_store


if __name__ == "__main__":
    print("Starting ingestion pipeline...\n")
    docs = load_documents()
    if not docs:
        print("No documents found in /data. Add some .txt files and re-run.")
        exit(1)

    chunks = split_documents(docs)
    build_vector_store(chunks)
    print("\nIngestion complete! You can now run the chatbot with: python app.py")
