# list_products.py
import os
from supabase import create_client, Client  
from dotenv import load_dotenv  

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
sb: Client = create_client(url, key)

def list_products():
    resp = sb.table("products").select("*").execute()
    return resp.data

if __name__ == "__main__":
    products = list_products()
    if products:
        print("Products List:")
        for p in products:
            print(f"ID: {p.get('product_id')}, Name: {p.get('name')}, SKU: {p.get('sku')}, Price: {p.get('price')}, Stock: {p.get('stock')}")
    else:
        print("No products found.")
