🛒 Farm2Bag Chat Assistant – AI-Powered E-commerce Support
An intelligent, AI-integrated e-commerce assistant that can:

Manage shopping carts 🛍️
Answer user questions 🤖
Understand natural language queries using LLMs 🧠
Recommend products 💡
Support voice input 🎤 and voice response 🔊
Dynamically respond to queries using RAG (Retrieval-Augmented Generation)

🚀 Tech Stack
💻 Frontend
HTML/CSS/JavaScript – Light UI for chat
Speech Recognition API – Voice input
Speech Synthesis API – Voice output
Fetch API – Sends and receives data to Flask backend
🧠 Backend
Python + Flask – REST API & web server
Flask-CORS – Enable cross-origin frontend-backend communication
MongoDB – Store products, cart data, and chat history
LangChain – Orchestrate LLMs, embedding, retrieval, prompts
Together AI – Used for embedding via TogetherEmbeddings
Groq Cloud + DeepSeek-LLaMA – Superfast hosted LLM inference
FAISS – Vector store for document retrieval (RAG)
BeautifulSoup – For scraping web content
Regex – Fallback intent recognition

 Key Features
🤖 AI Assistant
Uses Groq-hosted DeepSeek LLM for generating accurate, fast responses.
Extracts intent and entities using a language model.
Supports fallback pattern matching for better intent coverage.
📦 Smart Cart System
Add, update, remove, view cart items.
MongoDB stores product and cart info.
Shows dynamic pricing and totals.
🔍 RAG Integration
Uses Together Embeddings to convert content to vectors.
FAISS vectorstore enables semantic search.
If user’s intent is unknown, triggers RAG-based document Q&A.
🎙️ Voice Support
Accepts voice input using webkitSpeechRecognition.
Uses speechSynthesis to read chatbot responses aloud.

🧠 NLP + AI Stack

Intent & entity extraction	llm.invoke(prompt) (DeepSeek via Groq)
Embedding generation	TogetherEmbeddings
Vector search	FAISS (via LangChain)
LLM answer generation	ChatGroq using deepseek-r1-distill-llama-70b
Prompt templates	ChatPromptTemplate
Session memory	ChatMessageHistory (LangChain memory module)

🧪 Sample Intents Recognized
add_to_cart → “Add 2 mangoes to my cart”
show_cart → “What’s in my cart?”
update_cart → “Update bananas to 5”
remove_from_cart → “Remove onions”
recommend_items → “Suggest something healthy”
get_price → “How much is 1kg rice?”
If the model fails to understand → triggers RAG fallback with vector retrieval.

Local Set Up
Create a Virtual Environment: python -m venv venv
Install the requirements: pip install -r requirements.txt
Set up .env file:
GROQ_API_KEY=your_groq_api_key
TOGETHER_API_KEY=your_together_api_key
OPENAI_API_KEY=optional_if_used
Move in to directory: cd.\Farm2bag\
Run the backend: python app1.py
App runs locally on http://localhost:5000

<img width="1166" height="593" alt="f3" src="https://github.com/user-attachments/assets/4c327071-bac7-4520-a47c-d307cde8529d" />
<img width="1162" height="621" alt="f2" src="https://github.com/user-attachments/assets/60234fde-03f7-4f11-9375-e989a8882d7e" />
<img width="1163" height="622" alt="f1" src="https://github.com/user-attachments/assets/c035e401-3d23-4930-ac50-984c6dccafbf" />
