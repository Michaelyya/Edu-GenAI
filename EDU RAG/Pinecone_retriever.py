import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

def get_retriever():
    # Initialize Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    pc = Pinecone(api_key=pinecone_api_key)

    embedder = OpenAIEmbeddings()
    vectorstore = LangchainPinecone.from_existing_index(index_name=pinecone_index_name, embedding=embedder)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return retriever

if __name__ == "__main__":
    get_retriever()