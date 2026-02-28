import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

# Ensure .env variables are loaded properly
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

class QdrantManager:
    def __init__(self, collection_name="maritime_news"):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if qdrant_url and qdrant_api_key:
            print(f"🔗 Connecting to Qdrant Cloud at {qdrant_url}...")
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            # Fallback to local disk mode if cloud credentials aren't set
            qdrant_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_storage")
            os.makedirs(qdrant_path, exist_ok=True)
            print(f"🔗 Connecting to Local Qdrant Storage at {qdrant_path}...")
            self.client = QdrantClient(path=qdrant_path)
            
        self.collection_name = collection_name
        
        # The sentence-transformers all-MiniLM-L6-v2 model outputs vectors of size 384
        self.vector_size = 384 

        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Creates the collection if it doesn't already exist."""
        # Note: Depending on qdrant-client version, collection_exists might be available.
        # Otherwise we can get all collections and check.
        try:
            collections_response = self.client.get_collections()
            collection_names = [col.name for col in collections_response.collections]
            
            if self.collection_name not in collection_names:
                print(f"🛠️ Creating new Qdrant collection: '{self.collection_name}' with dimension {self.vector_size}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
            else:
                print(f"✅ Collection '{self.collection_name}' already exists.")
        except Exception as e:
            print(f"⚠️ Error checking/creating collection: {e}")

    def insert_chunks(self, points):
        """Inserts a list of PointStructs into Qdrant."""
        if not points:
            return
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"💾 Successfully inserted {len(points)} vector chunks into Qdrant.")

    def search(self, query_vector, limit=3):
        """Searches Qdrant for the most similar chunks to the query_vector."""
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        ).points
        return search_result
