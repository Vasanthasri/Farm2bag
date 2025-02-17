from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import re
import pymongo
import requests
from bs4 import BeautifulSoup
import groq
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_together import TogetherEmbeddings
from langchain_groq import ChatGroq
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

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
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Function to Add Items to Cart
def add_to_cart(user_id, item_name, quantity):
    item = products_collection.find_one({"name": {"$regex": f"^{re.escape(item_name)}$", "$options": "i"}})
    if not item:
        return f"❌ Item '{item_name}' not found."
    
    price = item["price"]
    cart_collection.insert_one({"user_id": user_id, "item_name": item_name, "quantity": quantity, "price": price})
    return f"🛒 Added {item_name} (Quantity: {quantity}) at ₹{price} per unit."

# Function to Remove Item from Cart
def remove_from_cart(user_id, item_name):
    result = cart_collection.delete_one({"user_id": user_id, "item_name": {"$regex": f"^{re.escape(item_name)}$", "$options": "i"}})
    if result.deleted_count > 0:
        return f"✅ Removed {item_name} from your cart."
    else:
        return f"❌ {item_name} is not in your cart."

# Function to Show Cart
def show_cart(user_id):
    cart_items = list(cart_collection.find({"user_id": user_id}, {"_id": 0, "item_name": 1, "quantity": 1, "price": 1}))
    if not cart_items:
        return "🛒 Your cart is empty."

    response = "**🛍️ Your Cart:**\n\n"
    for item in cart_items:
        response += f"🔹 {item['quantity']} x {item['item_name']} - ₹{item['price']} each\n"

    return response

# Function to Update Cart Item Quantity
def update_cart_item(user_id, item_name, new_quantity):
    item = cart_collection.find_one({"user_id": user_id, "item_name": {"$regex": f"^{re.escape(item_name)}$", "$options": "i"}})
    if item:
        price_per_unit = item.get("price", 0)
        new_price = new_quantity * price_per_unit
        cart_collection.update_one(
            {"user_id": user_id, "item_name": item["item_name"]},
            {"$set": {"quantity": new_quantity, "price": new_price}}
        )
        return f"✅ Updated {item['item_name']} to {new_quantity} units in your cart."
    return f"❌ {item_name} is not in your cart."

# Function to Recommend Products
def recommend_products(user_id):
    user_cart_items = list(cart_collection.find({"user_id": user_id}))
    cart_product_names = [item["item_name"] for item in user_cart_items]
    all_products = list(products_collection.find({}))

    recommended_items = []
    for product in all_products:
        if product["name"] not in cart_product_names:
            recommended_items.append(product["name"])
        if len(recommended_items) >= 5:
            break
    
    return recommended_items if recommended_items else ["Organic Rice", "Cold-Pressed Oil"]

# Function to Get Product Category
def get_product_category(product_name):
    product = products_collection.find_one({"name": product_name})
    return product.get("category", "") if product else ""

# Function to Get Price
def get_price(item_name):
    product = products_collection.find_one({"name": item_name})
    return product["price"] if product else 0

# Function to Get Cart Items
def get_cart_items(user_id):
    cart_items = list(cart_collection.find({"user_id": user_id}, {"_id": 0, "item_name": 1, "quantity": 1}))
    return cart_items

# Function to Extract Intent and Entities
def extract_intent_and_entities(user_message):
    """
    Extracts intent and entities (item_name, quantity) from the user's message using regex.
    """
    user_message = user_message.lower()

    # Patterns for different intents
    add_to_cart_pattern = re.compile(r"add\s+(\d+)?\s*([a-zA-Z\s]+)\s+to\s+(my\s+)?cart")
    remove_from_cart_pattern = re.compile(r"remove\s+([a-zA-Z\s]+)\s+from\s+(my\s+)?cart")
    update_cart_pattern = re.compile(r"update\s+([a-zA-Z\s]+)\s+to\s+(\d+)\s*(kg|pcs)?")
    show_cart_pattern = re.compile(r"show\s+(my\s+)?cart")
    recommend_items_pattern = re.compile(r"recommend\s+(me\s+)?(some\s+)?items?")
    get_price_pattern = re.compile(r"(price|cost)\s+of\s+([a-zA-Z\s]+)")

    # Check for add_to_cart intent
    match = add_to_cart_pattern.search(user_message)
    if match:
        quantity = int(match.group(1)) if match.group(1) else 1
        item_name = match.group(2).strip()
        return {"intent": "add_to_cart", "item": item_name, "quantity": quantity}

    # Check for remove_from_cart intent
    match = remove_from_cart_pattern.search(user_message)
    if match:
        item_name = match.group(1).strip()
        return {"intent": "remove_from_cart", "item": item_name}

    # Check for update_cart intent
    match = update_cart_pattern.search(user_message)
    if match:
        item_name = match.group(1).strip()
        new_quantity = int(match.group(2))
        return {"intent": "update_cart", "item": item_name, "quantity": new_quantity}

    # Check for show_cart intent
    if show_cart_pattern.search(user_message):
        return {"intent": "show_cart"}

    # Check for recommend_items intent
    if recommend_items_pattern.search(user_message):
        return {"intent": "recommend_items"}

    # Check for get_price intent
    match = get_price_pattern.search(user_message)
    if match:
        item_name = match.group(2).strip()
        return {"intent": "get_price", "item": item_name}

    # Default to unknown intent
    return {"intent": "unknown"}

# Serve Chatbot UI
@app.route('/chat')
def chat_page():
    return render_template("index.html")

# API Endpoint for Chat
@app.route('/process_chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    user_id = "user123"

    # Extract intent and entities
    extracted_data = extract_intent_and_entities(user_message)
    intent = extracted_data.get("intent")
    item_name = extracted_data.get("item", "").strip()
    quantity = extracted_data.get("quantity")

    # Handle intents
    if intent == "show_cart":
        response = show_cart(user_id)
    elif intent == "recommend_items":
        recommended_items = recommend_products(user_id)
        response = "🌟 You might also like: " + ", ".join(recommended_items)
    elif intent == "add_to_cart" and item_name:
        response = add_to_cart(user_id, item_name, quantity)
    elif intent == "remove_from_cart" and item_name:
        response = remove_from_cart(user_id, item_name)
    elif intent == "update_cart" and item_name and quantity:
        response = update_cart_item(user_id, item_name, quantity)
    elif intent == "get_price" and item_name:
        price = get_price(item_name)
        response = f"💰 The price of {item_name} is ₹{price} per unit." if price else f"❌ Sorry, {item_name} is not available."
    else:
        response = "❓ Sorry, I didn't understand. Can you rephrase?"

    return jsonify({"response": response})

# Run Flask Server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)