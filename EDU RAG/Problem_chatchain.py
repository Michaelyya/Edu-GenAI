import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from gpt_module import get_gpt_response, generate_prompt
from openai import OpenAI as GPTClient
import pinecone
import time
load_dotenv()

# Initialize Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = "us-east-1"
pinecone_index_name = "edu-research"
index_name = "edu-research"

# Initialize retriever and vectorstore
embedder = OpenAIEmbeddings()
pc = Pinecone(api_key=pinecone_api_key)
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)


if pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=pinecone_index_name, 
            dimension=1536, 
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
time.sleep(1)

existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

if pinecone_index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        pinecone_index_name,
        dimension=384,  # dimensionality of minilm
        metric='dotproduct',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pc.Index(pinecone_index_name)

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

def get_similar_docs(query, k, score=False):
    """Retrieve similar documents based on the input query."""
    query_vector = embedder.embed_query(query)
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    similar_docs = results['matches']
    results = [doc['metadata'] for doc in similar_docs]
    return results

# Initialize GPT Client
client = GPTClient(api_key=os.environ.get("OPENAI_API_KEY"))

def query_pinecone_problem(query, dok_level):
    """Process the query using Pinecone and generate a response using GPT."""
    similar_docs = get_similar_docs(query, k=3, score=True)
    prompt = generate_prompt(similar_docs, dok_level)
    response = get_gpt_response(prompt, client)
    return response


if __name__ == "__main__":
    query = input("Enter your query: ")
    dok_level = int(input("Enter the Depth of Knowledge (DOK) level (1-4): "))
    response = query_pinecone_problem(query, dok_level)
    print(response)
