from openai import OpenAI
from pymongo import errors, MongoClient


def get_mongodb_client(mongo_uri) -> MongoClient:
    """Establish connection to the MongoDB."""
    try:
        client = MongoClient(mongo_uri)
        print("Connection to MongoDB successful")
        return client
    except errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None


def get_embedding(client: OpenAI, text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    temp = client.embeddings.create(input=[text], model=model)
    return temp.data[0].embedding


def embed_documents(client: OpenAI, texts: list[str], model="text-embedding-3-small"):
    return [get_embedding(client, text, model) for text in texts]
