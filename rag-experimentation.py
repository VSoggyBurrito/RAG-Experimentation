import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from tqdm import tqdm

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
DATASET_PATH = r"filepath"
MODELS = {
    "sbert1": "all-MiniLM-L6-v2",
    "sbert2": "multi-qa-MiniLM-L6-cos-v1"
}
EMBEDDING_DIM = 384

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(data)} records")
        print("Sample record:", data[0] if data else "None")
    return [item["text"] for item in data if "text" in item]

def embed_texts(texts, model_name):
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True).tolist()

def setup_collections(client):
    for name in MODELS.keys():
        collection = f"{name}_collection"
        if not client.collection_exists(collection):
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )

def index_embeddings(client, name, embeddings, texts):
    client.upload_points(
        collection_name=f"{name}_collection",
        points=[
            PointStruct(id=i, vector=emb, payload={"text": txt})
            for i, (emb, txt) in enumerate(zip(embeddings, texts))
        ]
    )

def search(client, name, query_embedding, top_k=3):
    result = client.search(
        collection_name=f"{name}_collection",
        query_vector=query_embedding,
        limit=top_k
    )
    return [hit.payload["text"] for hit in result]

def main():
    texts = load_dataset(DATASET_PATH)
    if not texts:
        print("Dataset is empty or invalid.")
        return

    print("Connecting to Qdrant...")
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    setup_collections(client)

    for model_id, model_name in MODELS.items():
        print(f"\nEmbedding with {model_name}...")
        embeddings = embed_texts(texts, model_name)
        index_embeddings(client, model_id, embeddings, texts)

    while True:
        query = input("\nEnter a query (or type 'exit'): ")
        if query.lower() == "exit":
            break

        for model_id, model_name in MODELS.items():
            print(f"\nSearching with {model_name}...")
            query_embedding = embed_texts([query], model_name)[0]
            results = search(client, model_id, query_embedding)
            print(f"--- {model_id} Results ---")
            for res in results:
                print("•", res)

if __name__ == "__main__":
    main()
