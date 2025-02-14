from flask import Flask, request, jsonify
from flask_cors import CORS
import groq
import os
import re
import pymongo
from dotenv import load_dotenv
from bson.objectid import ObjectId
from pymongo import MongoClient

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = groq.Client(api_key=api_key)

# MongoDB Setup
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["farm2bag"]
products_collection = db["products"]
cart_collection = db["cart"]

# Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# ✅ Root Route
@app.route("/")
def home():
    return "Farm2Bag Chatbot Backend is running!"

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

# ✅ Function to recommend products based on cart items
def recommend_products(user_id):
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

# ✅ Function to determine similar or complementary products
def is_similar_or_complementary(product_name, cart_item):
    product_category = get_product_category(product_name)
    cart_item_category = get_product_category(cart_item)
    
    if product_category == cart_item_category:
        return True
    
    if cart_item.lower() in ["oranges", "apples"] and "juice" in product_name.lower():
        return True
    
    return False

# ✅ Endpoint to handle chatbot messages
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id", "user123")  # Replace with actual user authentication logic
    message = data.get("message", "")

    # Handle adding items to cart
    add_to_cart_pattern = re.compile(r"add\s+(\d+)?\s*([a-zA-Z\s]+)\s+to\s+my\s+cart", re.IGNORECASE)
    match = add_to_cart_pattern.search(message)

    if match:
        quantity = int(match.group(1)) if match.group(1) else 1
        item_name = match.group(2).strip()
        response = add_to_cart(user_id, item_name, quantity)
        return jsonify({"response": response})

    # Handle other queries (e.g., product recommendations)
    recommended_items = recommend_products(user_id)
    if recommended_items:
        response = "🌟 You might also like:\n" + "\n".join(recommended_items)
    else:
        response = "How can I assist you further?"

    return jsonify({"response": response})

# Run the Flask app
if __name__ == "__main__":
    app.run(port=5000)