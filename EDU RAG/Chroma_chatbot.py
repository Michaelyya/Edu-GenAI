import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from gpt_module import get_gpt_response, generate_prompt
from create_retriever import create_retriever, load_data
from openai import OpenAI
import os
dotenv.load_dotenv()
from langchain.agents import (
    create_openai_functions_agent,
    Tool,
    AgentExecutor,
)
from langchain import hub
from pinecone import Pinecone

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY']
)

REVIEWS_CHROMA_PATH = "chroma_data/"

reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=OpenAIEmbeddings()
)

docs = load_data("EDU RAG/examples/Math 140 Tutorial 3 Solutions.pdf")
reviews_retriever  = reviews_vector_db.as_retriever(k=10)
class ChatChain:
    def __init__(self, chat_model):
        self.retriever = create_retriever(docs)
        self.chat_model = chat_model

    def run(self, user_input, dok_level):
        # Retrieve relevant documents based on user input
        results = self.retriever.similarity_search(user_input, k=1)
        if results:
            retrieved_content = results[0].page_content  # Assume we take the most relevant document
        else:
            return "No relevant documents found."

        # Generate a response based on the retrieved content
        prompt = generate_prompt(retrieved_content, dok_level)
        response = get_gpt_response(prompt)
        return response


if __name__ == "__main__":
    chat_chain = ChatChain(chat_model = client)
    # Example user input
    user_input = "I need information on mathematical continuity principles."

    # Run the chat chain
    response = chat_chain.run(user_input=user_input, dok_level=2)
    print(response)