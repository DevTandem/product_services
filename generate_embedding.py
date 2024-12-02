from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os
import re
import json
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("Mongo_URL")
DB_NAME = os.getenv("DB_NAME")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
products_collection = db["products"]



app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')  

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5  


def generate_query_embedding(query):
    return model.encode(query).tolist()


def extract_price_constraints(query):
    match = re.search(r'(under|below|above|over)\s*(\d+)', query)
    if match:
        direction = match.group(1)
        amount = int(match.group(2))
        return direction, amount
    return None, None


def search_products(user_query, top_k=5):
    direction, amount = extract_price_constraints(user_query)
    textual_query = re.sub(r'(under|below|above|over)\s*\d+', '', user_query).strip()

    query_embedding = generate_query_embedding(textual_query)

    pipeline = [
        {
            "$search": {
                "index": "embedding",
                "knnBeta": {
                    "vector": query_embedding,
                    "path": "embedding",
                    "k": top_k
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "c_name": 1,
                "s_name": 1,
                "description": 1,
                "price": 1,
                "colour": 1,
                "characteristics": 1,
                "score": {"$meta": "searchScore"}
            }
        }
    ]

    results = list(products_collection.aggregate(pipeline))

    if direction and amount:
        if direction in ["under", "below"]:
            results = [res for res in results if float(res.get("price", float('inf'))) < amount]
        elif direction in ["above", "over"]:
            results = [res for res in results if float(res.get("price", 0)) > amount]

    return results


@app.post("/search_products/")
async def search(query: SearchQuery):
    try:
        results = search_products(query.query, query.top_k)
        return {"products": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
class EmbeddingRequest(BaseModel):
    text: str

@app.post("/generate_embedding/")
async def generate_embedding(request: EmbeddingRequest):
    try:
        embedding = model.encode(request.text).tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))