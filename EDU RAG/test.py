import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
load_dotenv()
from pinecone import Pinecone
api_key = os.environ.get("PINECONE_API_KEY")

print(f"Using API key: {api_key}")
# configure client
pc = Pinecone(api_key=api_key)

from pinecone import ServerlessSpec

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "edu-research"

import time

existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]
embedder = OpenAIEmbeddings()
# check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=384,  # dimensionality of minilm
        metric='dotproduct',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
print(index.describe_index_stats())

query = " Intermediate Value Theorem?"
xq = embedder.embed_query(query)
xc = index.query(vector=xq, top_k=3, include_metadata=True)
print(xc)