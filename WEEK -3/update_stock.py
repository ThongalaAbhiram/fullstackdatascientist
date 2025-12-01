# update_stock.py
import os
from supabase import create_client, Client  
from dotenv import load_dotenv  

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
sb: Client = create_client(url, key)

def update_stock(sku, new_stock):
    
    resp = sb.table("products").update({"stock": new_stock}).eq("sku", sku).execute()
    return resp.data

if __name__ == "__main__":
    sku = input("Enter SKU of product to update: ").strip()
    new_stock = int(input("Enter new stock quantity: ").strip())

    updated = update_stock(sku, new_stock)
    if updated:
        print("Updated product stock:", updated)
    else:
        print("No product found with that SKU.")
