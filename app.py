from flask import Flask, request, jsonify, render_template
from rag_chain import RAGChatbot

app = Flask(__name__)

# Load the RAG chatbot ONCE when the server starts (not on every request).
# This loads the vector store + connects to the local LLM.
print("Loading RAG chatbot... (this may take a few seconds)")
bot = RAGChatbot()
print("RAG chatbot loaded successfully.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        result = bot.ask(question)
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
        })
    except Exception as e:
        # In production you'd log this properly; here we just surface it for debugging
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
