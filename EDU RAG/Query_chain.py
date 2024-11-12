from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import os
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()
from Problem_chatchain import query_pinecone_problem
from Tutorial_chatchain import query_pinecone_tutorial

api_key=os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)


def find_route(question):
    class RouteQuery(BaseModel):
        """Route a user query to the most relevant datasource."""

        datasource: Literal["problem_vectorstore", "tutorial_vectorstore"] = Field(
            ...,
            description="Given a user question, choose to route it to problem vectorstore which contains examples of problem sets OR a tutorial vectorstore which contains definitions and knowledge set of problems",
        )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
    
    system = """You are a router. Given a user question, return the most relevant data source to answer the question. 
    We have two data sources:
    1. problem_vectorstore: This contains examples of problem sets. Choose this data source if the query is asking to generate a question or problem. 
       Example: "create a math problem involving algebra?"
    2. tutorial_vectorstore: This contains definitions and knowledge sets of problems. Choose this data source if the query is related to understanding definitions or concepts. 
       Example: "What is the definition of an algebraic expression?"
    3. prompting the answer directly: If you are unsure, ask the user to rewrite the question and say please refer math-related question.
    Carefully choose the data source to route the question to based on these criteria."""

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: {question}"),
        ]
    )
    
    route_chain = route_prompt | llm
    response = route_chain.invoke({"question": question})
    return response

def query_pinecone(question, dok_level,route):
    route = find_route(question).content.strip()
    print(route)
    if route == "Data source: problem_vectorstore":
        response = query_pinecone_problem(question, dok_level)
    elif route == "Data source: tutorial_vectorstore":
        response = query_pinecone_tutorial(question, dok_level)
    else:
        response = route
    return response

def question_answerable(question):
    return


if __name__ == "__main__":
    question = input("Enter your query: ")
    dok_level = int(input("Enter the Depth of Knowledge (DOK) level (1-4): "))
    route = find_route(question)
    response = query_pinecone(question, dok_level,route)
    print(response)


