
🛒 Farm2Bag Chat Assistant – AI-Powered E-commerce Support
An intelligent, AI-integrated e-commerce assistant that can:


-  Manage shopping carts  
-  Answer user questions  
-  Understand natural language queries using LLMs  
-  Recommend products  
-  Support voice input  
-  Provide voice responses  
---

## 🚀 Tech Stack


### 🖥️ Frontend
- HTML/CSS/JavaScript – Lightweight UI
- Speech Recognition API – Voice input
- Speech Synthesis API – Voice output
- Fetch API – Frontend-backend communication

  
### 🧠 Backend
- Python + Flask – Web server and REST API  
- Flask-CORS – Frontend-backend communication  
- MongoDB – Stores product/cart/chat data  
- LangChain – LLM orchestration  
- Together AI – Embedding via `TogetherEmbeddings`  
- Groq Cloud + DeepSeek-LLaMA – Fast LLM inference  
- FAISS – Vector store for semantic search  
- BeautifulSoup – Scraping web content  
- Regex – Fallback intent detection
---

## 🔑 Key Features


### 🤖 AI Assistant
- Utilizes Groq-hosted DeepSeek LLM for fast, accurate answers  
- Extracts user **intents** and **entities** using language models  
- Falls back to regex-based pattern recognition for unmatched inputs  

### 📦 Smart Cart System
- Add, update, delete, and view cart items  
- Stores product and cart info in MongoDB  
- Displays real-time totals and pricing  

### 🔍 RAG Integration (Retrieval-Augmented Generation)
- Uses Together Embeddings to convert documents into vector form  
- FAISS enables high-speed semantic search  
- Falls back to document-based Q&A for unrecognized queries  

### 🎙️ Voice Support
- Accepts voice input using `webkitSpeechRecognition`  
- Responds using `speechSynthesis` for a hands-free chat experience  

---


🧠 NLP + AI Stack

| Feature                    | Technology                                       |
| -------------------------- | ------------------------------------------------ |
| Intent & Entity Extraction | `llm.invoke(prompt)` using DeepSeek via Groq     |
| Embedding Generation       | `TogetherEmbeddings`                             |
| Vector Search              | `FAISS` via LangChain                            |
| Answer Generation          | `ChatGroq` using `deepseek-r1-distill-llama-70b` |
| Prompt Handling            | `ChatPromptTemplate`                             |
| Memory                     | `ChatMessageHistory` (LangChain session memory)  |
---


## 🧪 Sample Intents Recognized


- `add_to_cart` → “Add 2 mangoes to my cart”  
- `show_cart` → “What’s in my cart?”  
- `update_cart` → “Update bananas to 5”  
- `remove_from_cart` → “Remove onions”  
- `recommend_items` → “Suggest something healthy”  
- `get_price` → “How much is 1kg rice?”  
- ❓ Unknown intents → Triggers RAG fallback with vector-based document search  

---

## 🛠️ Local Setup Instructions


    ```bash
    # 1. Create a virtual environment
    python -m venv venv
    
    # 2. Activate the virtual environment (Windows)
    venv\Scripts\activate
    
    # 3. Install all required dependencies
    pip install -r requirements.txt
    
    # 4. Create a .env file and add the following:
    GROQ_API_KEY=your_groq_api_key
    TOGETHER_API_KEY=your_together_api_key
    OPENAI_API_KEY=optional_if_used
    
    # 5. Move into the project directory
    cd ./Farm2bag
    
    # 6. Run the Flask server
    python app1.py
    
    # The application will run at:
    http://localhost:5000


<img width="1166" height="593" alt="f3" src="https://github.com/user-attachments/assets/4c327071-bac7-4520-a47c-d307cde8529d" />
<img width="1162" height="621" alt="f2" src="https://github.com/user-attachments/assets/60234fde-03f7-4f11-9375-e989a8882d7e" />
<img width="1163" height="622" alt="f1" src="https://github.com/user-attachments/assets/c035e401-3d23-4930-ac50-984c6dccafbf" />
