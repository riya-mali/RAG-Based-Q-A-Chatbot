# 🤖 RAG Based AI Chatbot

A smart AI chatbot built using Retrieval-Augmented Generation (RAG) that provides accurate answers based on custom data instead of general internet knowledge.

---

## 🚀 Features

- ✅ Context-based question answering
- ✅ Uses custom documents (HR policies, FAQs, AI concepts)
- ✅ Reduces hallucination using RAG
- ✅ Fast retrieval using vector database
- ✅ Clean and interactive chat UI

---

## 🧠 Tech Stack

- Python
- LangChain
- Ollama (Llama3 Model)
- ChromaDB (Vector Database)
- Flask (Backend)
- HTML, CSS, JavaScript (Frontend)

---

## ⚙️ How It Works

1. Documents are stored in the `/data` folder  
2. Data is split into smaller chunks  
3. Converted into embeddings using Ollama  
4. Stored in ChromaDB (vector database)  
5. User asks a question  
6. Relevant data is retrieved  
7. LLM (Llama3.2) generates final answer

## 💬 Example Questions

- What is cloudsync?
- What is the pricing?
- What is reimbursement policy?
- What is CloudSync?
