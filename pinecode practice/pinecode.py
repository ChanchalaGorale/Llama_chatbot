import pinecone

from sentence_transformers import SentenceTransformer


# Project: Semantic search project


key="5d288893-ff54-496b-a94b-3fef7233d413"

pinecone.init(api_key=key)

index_name="quickstart"

# create index
pinecone.create_index(index_name, dimensions=768)

# connect with index
index = pinecone.Index(index_name)

# Load a pre-trained Sentence Transformer model 
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# dataset

documents = ["Document 1 content", "Document 2 content", "Document 3 content"]

vectors= model.encode(documents)


# store all vectors in pinecone
for i, vec in enumerate(vectors):
    index.upsert([(f"doc{i}", vec)])


query = "Content related to document 1"

query_vec = model.encode([query])[0]

search_res= index.query([query_vec], top_k=5)










