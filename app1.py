import chainlit as cl
import groq
import os
import json
import re
import PyPDF2
import pymongo
from dotenv import load_dotenv
from bson.objectid import ObjectId
from pymongo import MongoClient
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
from datetime import datetime

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = groq.Client(api_key=api_key)
os.environ["TOGETHER_API_KEY"] = 'c97530aceaffcb28eecbaefd032551e2075fd3d323aea71f598cad92122c7d69'
store = {}

# MongoDB Setup
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["farm2bag"]
products_collection = db["products"]
cart_collection = db["cart"]

# Vector Embeddings Model
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

# LLM Model
llm1 = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# ✅ Function to Add Items to Cart
def add_to_cart(user_id, item_name, quantity):
    """Adds an item to the cart or updates the quantity if it already exists."""
    existing_item = cart_collection.find_one({"user_id": user_id, "item_name": item_name})

    if existing_item:
        cart_collection.update_one(
            {"user_id": user_id, "item_name": item_name},
            {"$inc": {"quantity": quantity}}
        )
        return f"✅ Updated {item_name} quantity in your cart."
    else:
        cart_item = {
            "user_id": user_id,
            "item_name": item_name,
            "quantity": quantity
        }
        cart_collection.insert_one(cart_item)
        return f"🛒 Added {item_name} to your cart."

# ✅ Function to get product category
def get_product_category(product_name):
    product = products_collection.find_one({"name": product_name})
    return product.get("category", "") if product else ""

# ✅ Function to determine similar or complementary products
def is_similar_or_complementary(product_name, cart_item):
    product_category = get_product_category(product_name)
    cart_item_category = get_product_category(cart_item)
    
    if product_category == cart_item_category:
        return True
    
    if cart_item.lower() in ["oranges", "apples"] and "juice" in product_name.lower():
        return True
    
    return False

# ✅ Function to recommend products based on cart items
def recommend_products(user_id):
    """Recommends products based on items in the user's cart."""
    user_cart_items = list(cart_collection.find({"user_id": user_id}))
    cart_product_names = [item["item_name"] for item in user_cart_items]
    all_products = list(products_collection.find({}))
    
    recommended_items = []
    for product in all_products:
        if product["name"] in cart_product_names:
            continue
        
        for cart_item in cart_product_names:
            if is_similar_or_complementary(product["name"], cart_item):
                recommended_items.append(product["name"])
                break
    
    return recommended_items[:5] if recommended_items else ["Organic Rice", "Cold-Pressed Oil"]

def get_cart_items(user_id):
    cart_items = list(cart_collection.find({"user_id": user_id}, {"_id": 0, "item_name": 1, "quantity": 1}))
    return cart_items

def handle_user_input(user_id, user_input):
    user_input = user_input.lower()

    # Pattern to detect updates: (e.g., "update apples to 3 kg", "change bananas to 5 pcs")
    update_pattern = re.search(r"(update|change|modify|make) (.*?) to (\d+)\s?(kg|pcs)?", user_input)

    if update_pattern:
        item_name = update_pattern.group(2).strip()  # Extract the item name
        new_quantity = int(update_pattern.group(3))  # Extract the quantity

        # Call the update function
        update_message = update_cart_item(user_id, item_name, new_quantity)
        return update_message

    # If the user asks to show the cart
    elif "show" in user_input and "cart" in user_input:
        return show_cart(user_id)

    else:
        return "🤖 I didn't understand that. You can say things like 'Update Red Banana to 10 pcs' or 'Make oranges 2 kg'."
    
def remove_from_cart(user_id, item_name):
    """Removes an item from the cart."""
    result = cart_collection.delete_one({"user_id": user_id, "item_name": item_name})
    if result.deleted_count > 0:
        return f"✅ Removed {item_name} from your cart."
    else:
        return f"❌ {item_name} is not in your cart."

def show_cart(user_id):
    """Displays all cart items with their prices and the total cost."""
    cart_items = list(cart_collection.find({"user_id": user_id}, {"_id": 0, "item_name": 1, "quantity": 1}))

    if not cart_items:
        return "🛒 Your cart is currently empty."

    response = "**🛍️ Your Cart:**\n\n"
    subtotal = 0

    for item in cart_items:
        item_name = item["item_name"]
        quantity = item["quantity"]
        price_per_unit = get_price(item_name)
        total_price = price_per_unit * quantity
        subtotal += total_price

        response += f"🔹 {quantity} x {item_name} - ₹{price_per_unit} each = ₹{total_price}\n"

    response += f"\n**Total: ₹{subtotal}**"
    response += "\nWould you like to continue shopping or checkout?"
    return response

def get_price(item_name):
    """Fetches the price of an item from the products collection."""
    product = products_collection.find_one({"name": item_name})
    return product["price"] if product else 0

def update_cart(user_id, item_name, new_quantity):
    """Updates the quantity of an item in the cart."""
    # Normalize item name (to handle case and space inconsistencies)
    item_name = item_name.strip().lower()

    # Check if the item exists in the cart
    existing_item = cart_collection.find_one({"user_id": user_id, "item_name": item_name})

    if existing_item:
        # Update the quantity if the item is found
        cart_collection.update_one(
            {"user_id": user_id, "item_name": item_name},
            {"$set": {"quantity": new_quantity}}
        )
        return f"✅ Updated {item_name} to {new_quantity} kg(s) in your cart."
    else:
        return f"❌ {item_name} is not in your cart."
    
def get_cart_items(user_id):
    cart_items = list(cart_collection.find({"user_id": user_id}, {"_id": 0, "item_name": 1, "quantity": 1}))
    return cart_items


# ✅ Chatbot Welcome Message
@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome to Farm2Bag chatbot! How can I assist you today?").send()

# ✅ Handling User Messages
@cl.on_message
async def respond(message: cl.Message):
    user_id = "user123"  # Replace with actual user identification logic

    add_to_cart_pattern = re.compile(r"add\s+(\d+)?\s*([a-zA-Z\s]+)\s+to\s+my\s+cart", re.IGNORECASE)
    match = add_to_cart_pattern.search(message.content)

    if match:
        quantity = int(match.group(1)) if match.group(1) else 1
        item_name = match.group(2).strip()

        add_to_cart_response = add_to_cart(user_id, item_name, quantity)
        await cl.Message(content=add_to_cart_response).send()
        return

    # ✅ Remove from Cart
    remove_from_cart_pattern = re.compile(r"remove\s+([a-zA-Z\s]+)\s+from\s+my\s+cart", re.IGNORECASE)
    remove_match = remove_from_cart_pattern.search(message.content)

    if remove_match:
        item_name = remove_match.group(1).strip()
        remove_response = remove_from_cart(user_id, item_name)
        await cl.Message(content=remove_response).send()
        return

    # ✅ Update Cart Item Quantity
    update_cart_pattern = re.compile(r"update\s+([a-zA-Z\s]+)\s+to\s+(\d+)\s*(kg|pcs)?", re.IGNORECASE)
    update_match = update_cart_pattern.search(message.content)

    if update_match:
        item_name = update_match.group(1).strip()
        new_quantity = int(update_match.group(2))
        update_response = update_cart(user_id, item_name, new_quantity)  # Use update_cart here
        await cl.Message(content=update_response).send()
        return

    # ✅ Show Cart
    show_cart_pattern = re.compile(r"show\s+(my\s+)?cart", re.IGNORECASE)
    if show_cart_pattern.search(message.content):
        cart_response = show_cart(user_id)
        await cl.Message(content=cart_response).send()
        return

    # ✅ Recommend Items
    recommend_pattern = re.compile(r"recommend\s+(me\s+)?(some\s+)?items?", re.IGNORECASE)
    if recommend_pattern.search(message.content):
        recommended_items = recommend_products(user_id)
        if recommended_items:
            recommendations = "🌟 You might also like:\n" + "\n".join(recommended_items)
            await cl.Message(content=recommendations).send()
        else:
            await cl.Message(content="🌟 No recommendations available at the moment.").send()
        return

    # ✅ Load FAISS Vectorstore for RAG
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    # ✅ Create Chat Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Farm2Bag customer service assistant. {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # ✅ Create Retrieval-Augmented Generation (RAG) Chain
    question_answer_chain = create_stuff_documents_chain(llm1, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history, input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer"
    )

    # ✅ Generate Response from RAG
    response = conversational_rag_chain.invoke({"input": message.content}, config={"configurable": {"session_id": "abc123"}})["answer"]
    bot_reply = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    await cl.Message(content=bot_reply).send()


# ✅ Function to Get Chat Session History
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]