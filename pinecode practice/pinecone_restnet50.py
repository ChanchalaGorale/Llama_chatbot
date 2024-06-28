import pinecone
import torch
from PIL import Image 
from torchvision import models, transforms



key="pineconekey"


pinecone.init(api_key=key)


index_name= "some-index-name"
# create index

pinecone.create_index(index_name, dimentions=100)




# connect to index
index = pinecone.Index(index_name)


model= models.resnet50(pretrained=True)

model= model.eval()


# preprocess image

proprocess= transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.toTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])


# upsert vectors

index.upsert([("id", "vector")])


query_vector=["question vector"]

# query vectors

index.query(query_vector,top_k=3)
