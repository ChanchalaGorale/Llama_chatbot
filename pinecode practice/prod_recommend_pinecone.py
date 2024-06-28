import pinecone
import numpy as np

key="5d288893-ff54-496b-a94b-3fef7233d413"


pinecone.init(api_key=key)

index_name= "product-recommendation-system"


pinecone.create_index(index_name, dimentions=512)


index = pinecone.Index(index_name)

product_vectors= np.random.rand(100, 512)

product_ids=[f"product {i}" for i in range(100)]


index.upsert([(product_ids[i], product_vectors[i]) for i in range(len(product_vectors)) ])


user_vec =  np.random.rand(1, 512)


index.query(user_vec.tolist(), top_k=5)

