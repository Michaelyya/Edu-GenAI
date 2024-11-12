import dotenv
import openai
from langchain_community.document_loaders import PyPDFium2Loader
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()
REVIEWS_CHROMA_PATH = "chroma_data"

def load_data(path):
    loader = PyPDFium2Loader(path)
    docs = loader.load()
    return docs

def create_retriever(docs):
    embeddings = OpenAIEmbeddings()
    retriever = Chroma(persist_directory=REVIEWS_CHROMA_PATH).from_documents(
        docs,
        OpenAIEmbeddings(),
    )
    return retriever

def test_retrieval(retriever):
    test_query = "give me the solution for limx→πsin(x+ sinx)"
    results = retriever.similarity_search(test_query, k=3)
    print(results[0].page_content)

if __name__ == "__main__":
    path = "/Users/yonganyu/Desktop/handon RAG/EDU RAG/examples/Math 140 Tutorial 3 Solutions.pdf"
    docs = load_data(path)
    retriever = create_retriever(docs)
    test_retrieval(retriever)
