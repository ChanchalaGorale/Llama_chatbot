from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="5d288893-ff54-496b-a94b-3fef7233d413")

pc.create_index(
    name="quickstart",
    dimension=8, # Replace with your model dimensions
    metric="euclidean", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)