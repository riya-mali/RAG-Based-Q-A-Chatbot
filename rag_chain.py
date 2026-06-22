from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"   # local chat model served by Ollama

# The prompt template tells the LLM to ONLY answer using the retrieved context.
# This is what makes answers "grounded" instead of hallucinated.
PROMPT_TEMPLATE = """You are a helpful assistant answering questions based ONLY on the context provided below.
If the answer is not contained in the context, say "I don't have enough information to answer that."
Do not make up information that isn't in the context.

Context:
{context}

Question: {question}

Answer clearly and concisely:"""


def format_docs(docs):
    """Combine retrieved chunks into a single context string, with source labels."""
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


class RAGChatbot:
    def __init__(self):
        # Load the same embedding model used during ingestion
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)

        # Load the persisted vector store from disk
        self.vector_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings,
            collection_name="knowledge_base",
        )

        # Retriever: fetches top-k most similar chunks for a given query
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # The LLM that generates the final answer
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        # The full RAG chain, built using LangChain's pipe syntax:
        # question -> retrieve context -> fill prompt -> LLM -> parse to string
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str):
        """Run the RAG chain and also return which sources were used (for transparency)."""
        retrieved_docs = self.retriever.invoke(question)
        sources = sorted(set(doc.metadata.get("source", "unknown") for doc in retrieved_docs))

        answer = self.chain.invoke(question)

        return {
            "answer": answer,
            "sources": sources,
        }


if __name__ == "__main__":
    # Quick manual test from the command line
    bot = RAGChatbot()
    print("RAG Chatbot ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.lower() in ("exit", "quit"):
            break
        result = bot.ask(q)
        print(f"\nBot: {result['answer']}")
        print(f"(Sources: {result['sources']})\n")
