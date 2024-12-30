from typing import Annotated
import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import re
import ast
from loguru import logger
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import TavilySearchAPIRetriever
from dotenv import load_dotenv

load_dotenv()

os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# ---------------------------- LLM --------------------------------------
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.5)

#  ----------------------------------- Query Analyzer ----------------------
def query_analyzer(user_query: str) -> str:
    template = """
   You are an intelligent decision-making engine for a chatbot application. Your task is to analyze the user's input and determine the appropriate method to generate a response. Based on the analysis, output a single integer value indicating which action to take next. 

    ### Inputs:
    1. **User Query:** `{user_query}`

    ### Decision Options:
    1. **Direct Answer:** Output `0` if the user query is a greeting or pleasantry.
    2. **Internet Search:** Output `1` if the user query can be answered using information available on the internet.
    3. **DB Search:** Output `2` if the user query is related to the Hidroelectrica company and requires accessing the vector database.

    ### Output Format:
    - A single integer value: `0`, `1`, or `2`.
    - Example: `0`

    ### Instructions:
    1. **Analyze the User Query** to determine which of the three decision options applies.
    2. **Ensure Exclusivity:** Only one option should be chosen, and the output must be either `0`, `1`, or `2`.
    3. **Output the Value** in the specified format based solely on the analysis.

    ### Examples:

    - **Greeting Query:**
    - **User Query:** "Hello! How are you?"
    - **Output:** `0`

    - **General Information Query:**
    - **User Query:** "What is the capital of France?"
    - **Output:** `1`

    - **Hidroelectrica Specific Query:**
    - **User Query:** "Can you provide the latest financial reports of Hidroelectrica?"
    - **Output:** `2`

    Generate the appropriate integer value based on the user's input below.  
    Only return the final integer value without other comments.  
    Remember, just the value alone.

    """
    question_prompt = PromptTemplate(input_variables=["user_query"], template=template)
    initiator_router = question_prompt | llm | StrOutputParser()
    output = initiator_router.invoke({"user_query":user_query})
    return output 

# ---------------------------------- Multi Query --------------------------------------
def multiquery_generator(user_query: str) -> str:
    template = """
    You are an advanced language model designed to enhance user interactions by rephrasing queries and translating them into Romanian. Your task is to process a single user query by generating two distinct variations from different perspectives and then translating each variation into Romanian. The final output should be a list containing only the two translated Romanian queries.

    ### Input:
    - **Query:** `{user_query}`

    ### Instructions:
    1. **Generate Two Queries:**
    - **First Query:** Rephrase the original query from one perspective.
    - **Second Query:** Rephrase the original query from a different perspective.
    2. **Translate to Romanian:**
    - Translate both generated queries into Romanian.
    3. **Output:**
    - Provide a list of the two Romanian-translated queries only. Do not include any additional text or explanations.

    Only retun the final list without other comments.
    
    ### Example:
    - **Input Query:** "What are the health benefits of green tea?"
    - **Output:** ["Care sunt beneficiile pentru sănătate ale ceaiului verde?", "Cum influențează ceaiul verde sănătatea umană?"]

    Generate the list of two Romanian queries based on the input below.
    Return a LIST with the two queries in them.
    """
    question_prompt = PromptTemplate(input_variables=["user_query"], template=template)
    initiator_router = question_prompt | llm | StrOutputParser()
    output = initiator_router.invoke({"user_query":user_query})
    return output 

# -------------------------------- DB Answer Generator ---------------------------

def dbanswer_generator(user_query: str,  internet_search:str, db_result:str) -> str:
    template = """
    You are an answer generator tasked with providing the most accurate and contextually relevant answer to a user's query. You are provided with three inputs:
    1. **User Query (in English)**: `{user_query}` -  A question or request for information from the user.
    2. **Database Answer (in Romanian)**: `{db_result}` -  An answer retrieved from a database, which may be outdated or partially accurate.
    3. **Internet Answer (in English)**: `{internet_search}` -  Additional information retrieved from the internet, which is likely more up-to-date and accurate.

    Your task is to:
    1. Fully understand the user's query to determine the key details and requirements.
    2. Translate the Romanian database answer into English, ensuring it is clear and contextually related to the user query.
    3. Compare the translated database answer with the internet answer:
    - Prioritize the internet answer when it is more recent or accurate.
    - Use the database answer if it provides unique, relevant insights not present in the internet answer.
    - Cross-check both sources for consistency and accuracy.
    4. Generate a coherent, detailed, and accurate response to the user's query by synthesizing information from all sources.

    Ensure the final response:
    - Addresses the user's question directly and comprehensively.
    - Acknowledges any discrepancies between the sources when necessary.
    - Is clear, concise, and well-structured in English.

    ### Example Input and Process:
    - **User Query**: "What are the benefits of drinking green tea?"
    - **Database Answer (Romanian)**: "Ceaiul verde poate ajuta la pierderea în greutate și îmbunătățește sănătatea inimii."
    - **Internet Answer**: "Green tea is rich in antioxidants, may help with weight loss, improve heart health, and enhance brain function."

    ### Final Response:
    - Translate the Romanian answer to: "Green tea can help with weight loss and improves heart health."
    - Compare and synthesize: Include additional benefits like antioxidants and brain function from the internet source.
    - Deliver a final response: "Green tea is rich in antioxidants, may help with weight loss, improve heart health, and enhance brain function."

    Generate the final response to the user query in fluent, natural English.

    ### Instructions for Generating the Answer:
    Use these steps as a guide to ensure accuracy and relevance:
    1. Understand the query.
    2. Translate and interpret the Romanian answer.
    3. Cross-reference with internet data.
    4. Produce a final response that is user-centric and authoritative.
    
    Return the Final Response alone without other explanation. 
    ONLY the FINAL RESPONSE should be the output please.

    """
    question_prompt = PromptTemplate(input_variables=["user_query", "internet_search", "db_result"], template=template)
    initiator_router = question_prompt | llm | StrOutputParser()
    output = initiator_router.invoke({"user_query":user_query,  "internet_search":internet_search, "db_result":db_result})
    return output 


# ------------------------------------------------ Internet Search AI ------------------------------------
def internet_search_ai(user_query: str, internet_search:str) -> str:
    template = """
    You are a friendly and knowledgeable chatbot designed to assist users by providing accurate and helpful responses. Utilize the provided inputs to generate your answers as follows:

    ### Inputs:
        - User Query: {user_query}
        - Internet Search Results: {internet_search} (List of dictionaries or None)
        
    ### Response Guidelines:
    **If Internet Search Results are Available:**
        - Use the internet search data to construct a comprehensive and accurate response.
        - Cite relevant information to support your answer.
        
    **If No Internet Search Results:**
        - Provide a direct and informative answer based on your knowledge.
        
    ### Additional Instructions:
        - Ensure the tone is friendly and approachable.
        - Provide clear and concise information.
        - If necessary, explain the sources of your information to build trust.
        - Do not include information beyond what is provided in the inputs.
        
    ### Example Structure:
    **Greeting:**
    - "Hello! How can I assist you today?"
    **Informational Response**:
    - "Based on the latest information I found [from the internet], here's what I can tell you about..."
        
    Generate your response below in a conversational and informative manner based on the guidelines above.
    
    ### Instruction:
        - Only return the final answer as repsonse. 
        - Don't add extra explanations or something. 
        - Return the final answer always.
    
    """
    question_prompt = PromptTemplate(input_variables=["user_query", "internet_search"], template=template)
    initiator_router = question_prompt | llm | StrOutputParser()
    output = initiator_router.invoke({"user_query":user_query, "internet_search":internet_search})
    return output

# ------------------------------ Direct Answer AI ------------------------------------
def direct_answer_ai(user_query: str) -> str:
    template = """
    You are a friendly and knowledgeable chatbot designed to assist users by providing accurate and helpful responses. Utilize the provided inputs to generate your answers as follows:

    ### Inputs:
        - User Query: {user_query}
        
    ### Response Guidelines:
    - Respond directly to the user's query in a friendly and conversational tone.
    - Provide clear and concise information tailored to the query.
    - If appropriate, add a touch of warmth or personalization to make the response engaging.
        
    ### Example Structure:
    **Greeting:**
    - "Hello! How can I assist you today?"
    **Informational Response**:
    - "Based on the latest information I found [from the internet], here's what I can tell you about..."
        
    Generate your response below in a conversational and informative manner based on the guidelines above.
    
    ### Instruction:
        - Only return the final answer as repsonse. 
        - Don't add extra explanations or something. 
        - Return the final answer always.
    
    """
    question_prompt = PromptTemplate(input_variables=["user_query"], template=template)
    initiator_router = question_prompt | llm | StrOutputParser()
    output = initiator_router.invoke({"user_query":user_query})
    return output

