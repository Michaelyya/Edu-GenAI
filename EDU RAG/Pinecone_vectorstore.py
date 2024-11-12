import openai
import langchain
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone as LangchainPinecone


# Set up Pinecone

# Load environment variables from .env file
load_dotenv()
embed_model = "text-embedding-ada-002"

def vectorstore(path):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    pdf_texts = text_splitter.split_documents(pages)

    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

    pinecone_index_name = "edu-research"
    pinecone_environment = "us-east-1"

    print(
        f"Creating index: {pinecone_index_name} in {pinecone_environment} environment"
    )

    if pinecone_index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=pinecone_index_name,
            dimension=1536,  # Dimension should match the embedding size
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region=pinecone_environment
            )
        )
        print(f"Created index: {pinecone_index_name}")

    embedder = OpenAIEmbeddings()

    vectorstore = LangchainPinecone.from_documents(
        pdf_texts, embedder, index_name=pinecone_index_name
    )

    return vectorstore

if __name__ == "__main__":
    vectorstore_instance = vectorstore("EDU RAG/examples/Math 140 Tutorial 3 Solutions.pdf")