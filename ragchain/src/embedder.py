# Embedding

import uuid
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

def get_vector_store(
    embedding_model: str = "thenlper/gte-large",
    collection_name: str = "nigerian_edu",
    vector_size: int = 1024,
    distance: Distance = Distance.COSINE,
    client_url: str = ":memory:",
    api_key: str = None,
    device: str = "cpu",
    recreate: bool = False, # Set to True to when testing
) -> tuple[QdrantVectorStore, QdrantClient]:
    """
    Create and return a QdrantVectorStore and QdrantClient with specified configuration.
    Adds logging and error handling for visibility.
    """
    print(f"🔌 Initializing HuggingFaceEmbeddings on device: {device} ...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": device})

    try:
        if client_url == ":memory:":
            print("🗄️  Creating in-memory Qdrant client...")
            client = QdrantClient(":memory:")
        else:
            print(f"🗄️  Connecting to Qdrant at {client_url} ...")
            client = QdrantClient(url=client_url, api_key=api_key)
    except Exception as e:
        print(f"❌ Error creating Qdrant client: {e}")
        raise

    try:
        if recreate:
            print(f"⚠️  Recreating collection '{collection_name}' in Qdrant...")
            client.recreate_collection(
                collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )
        else:
            if not client.collection_exists(collection_name=collection_name):
                print(f"📦 Creating collection '{collection_name}' in Qdrant...")
                client.create_collection(
                    collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance)
                )
            else:
                print(f"✅ Collection '{collection_name}' already exists.")
    except Exception as e:
        print(f"❌ Error creating/recreating Qdrant collection: {e}")
        raise

    print("✅ Qdrant vector store and client ready.")
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
    return vector_store, client


def embed_documents(
    chunks: list[Document],
    vector_store: QdrantVectorStore,
    client: QdrantClient,
    collection_name: str = "nigerian_edu",
    batch_size: int = 300,
) -> None:
    """
    Embed document chunks into Qdrant vector store in batches.
    Adds a unique 'id' to each chunk metadata if missing.
    Prints confirmation message with total vectors stored.
    """

    # Add unique ID to each chunk metadata if not already present
    for doc in chunks:
        if "id" not in doc.metadata:
            doc.metadata["id"] = str(uuid.uuid4())

    import time
    # Batch embedding
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding in batches"):
        batch = chunks[i : i + batch_size]
        try:
            t0 = time.time()
            vector_store.add_documents(batch)
            print(f"Batch {i} embedded in {time.time() - t0:.2f}s")
        except Exception as e:
            print(f"Error embedding batch {i}: {e}")

    # Confirm total vectors count
    try:
        total_vectors = client.count(collection_name=collection_name).count
        print(f"\nEmbedding complete! Total vectors stored in Qdrant: {total_vectors}")
    except Exception as e:
        print(f"Embedding may be complete, but couldn't confirm count: {e}")
