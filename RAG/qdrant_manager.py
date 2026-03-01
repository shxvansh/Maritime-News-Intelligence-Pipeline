"""
QdrantManager - Production-Grade Vector Storage Interface
Enforces Qdrant Cloud (no local fallback). Supports Hybrid Search (Dense + Sparse).
"""

import os
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseVector
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


class QdrantManager:
    """
    Manages the Qdrant Cloud vector collection for the Maritime RAG pipeline.
    Supports Dense (BGE-large 1024-D) + Sparse (BM25) Hybrid Search.
    """

    # BGE-large-en-v1.5 outputs 1024-dimensional vectors
    DENSE_VECTOR_SIZE = 1024
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(self, collection_name="maritime_news_v2"):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise EnvironmentError(
                "❌ QDRANT_URL and QDRANT_API_KEY must be set in your .env file. "
                "Local storage is not supported in production. "
                "Sign up at https://cloud.qdrant.io to get your credentials."
            )

        print(f"🔗 Connecting to Qdrant Cloud at {qdrant_url}...")
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Creates the collection with Dense + Sparse vector support if it doesn't exist."""
        try:
            collections_response = self.client.get_collections()
            collection_names = [col.name for col in collections_response.collections]

            if self.collection_name not in collection_names:
                print(f"🛠️ Creating Qdrant collection: '{self.collection_name}'")
                print(f"   Dense Vector: {self.DENSE_VECTOR_SIZE}-D (Cosine)")
                print(f"   Sparse Vector: BM25 Keyword Index")

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        self.DENSE_VECTOR_NAME: VectorParams(
                            size=self.DENSE_VECTOR_SIZE,
                            distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        self.SPARSE_VECTOR_NAME: SparseVectorParams()
                    },
                )
                print(f"✅ Collection '{self.collection_name}' created successfully.")
            else:
                print(f"✅ Collection '{self.collection_name}' already exists.")
        except Exception as e:
            print(f"⚠️ Error checking/creating collection: {e}")
            raise

    def insert_chunks(self, points):
        """Inserts a list of PointStructs (with named dense + sparse vectors) into Qdrant."""
        if not points:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"💾 Inserted {len(points)} vector chunks into Qdrant Cloud.")

    def hybrid_search(self, dense_vector, sparse_vector, limit=5):
        """
        Performs Hybrid Search using Reciprocal Rank Fusion (RRF).
        Combines Dense (semantic meaning) + Sparse (exact keyword) results.
        """
        # Use Qdrant's native query API with prefetch for hybrid search
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # Dense semantic search pass
                models.Prefetch(
                    query=dense_vector,
                    using=self.DENSE_VECTOR_NAME,
                    limit=limit * 3,  # Over-fetch for better fusion
                ),
                # Sparse keyword search pass
                models.Prefetch(
                    query=sparse_vector,
                    using=self.SPARSE_VECTOR_NAME,
                    limit=limit * 3,
                ),
            ],
            # Reciprocal Rank Fusion to merge the two result sets
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        ).points

        return search_results

    def dense_search(self, dense_vector, limit=5):
        """Fallback: performs a pure dense vector search (no sparse component)."""
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using=self.DENSE_VECTOR_NAME,
            limit=limit,
        ).points
        return search_results
