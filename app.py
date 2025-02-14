import chainlit as cl
import groq
import os
from dotenv import load_dotenv
import re
import PyPDF2
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_together import TogetherEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Retrieve API Key from .env file
api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq client
client = groq.Client(api_key=api_key)
os.environ["TOGETHER_API_KEY"] = 'c97530aceaffcb28eecbaefd032551e2075fd3d323aea71f598cad92122c7d69'
store = {}


embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
)

llm1 = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

def extract_text(text_file):
    with open(text_file.path, 'rb') as file:  # Open in binary mode
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure that text was extracted
                text += page_text + "\n"
        return text

def vectorize_text(docs, k=10):  # Limit to top 10 documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(docs)
    # Create FAISS vectorstore
    vectorstore = FAISS.from_texts(texts=splits, embedding=embeddings)
    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome to Farm2Bag chatbot! How can I assist you today?").send()

@cl.on_message
async def respond(message: cl.Message):

    
    text = extract_text("scraped_products_details.pdf")
    retriever = vectorize_text(text)
    system_prompt = ("You are a farm2bag customer model."
                     "You are a customer service chatbot."
                     """Follow these guidelines:
1. STRICTLY use only information from the provided context
2. For numerical data, always include exact figures and their source page
3. Maintain original technical terminology from the document
4. Structure answers with: summary, key points, and document references
5. If uncertain, specify which aspects need clarification"""
                         "{context}")

    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    question_answer_chain = create_stuff_documents_chain(llm1, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    response = conversational_rag_chain.invoke(
            {"input": message.content},
            config={
                "configurable": {"session_id": "abc123"}
            },  # constructs a key "abc123" in `store`.
        )["answer"]

    await cl.Message(content = response).send()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
