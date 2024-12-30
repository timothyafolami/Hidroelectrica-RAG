import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
import re
import ast
from loguru import logger
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import TavilySearchAPIRetriever
from dotenv import load_dotenv

# ----------------------- Local Imports ---------------------------------
from src.ai_functions import *

load_dotenv()

os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


# ---------------------------- LLM --------------------------------------
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.5)


# ------------------------- DB Loader ---------------------------
logger.info("Loading Embedding model")
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"} #can also be cpu
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )

# Loading the vector database
vec_db = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)

# Minimal Data Cleaner
def remove_symbols(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return cleaned_text

#  Document Searcher
def most_similar_document(query: list) -> list:
    all_docs = []
    query = list(set(query))
    for doc in query:
        retrieval = vec_db.as_retriever(search_type="mmr", search_kwargs={"k": 1, 'lambda_mult': 0.25})
        query_docs = retriever.invoke(doc)
        docs = [remove_symbols(i.page_content) for i in query_docs[:1]]
        all_docs.extend(docs)
    return all_docs



# --------------------- Internet search -----------------------------------
search = DuckDuckGoSearchRun()
retriever = TavilySearchAPIRetriever(k=2)

def internet_searcher(query, k=2):
    """
    Function to perform a search using Tavily as the primary retriever,
    and falls back to DuckDuckGo if Tavily fails.
    
    Args:
    - query (str): The search query.
    - k (int): Number of results to retrieve with Tavily.

    Returns:
    - dict: Search results from Tavily or DuckDuckGo.
    """
    # Initialize Tavily and DuckDuckGo retrievers
    retriever = TavilySearchAPIRetriever(k=k)
    duck_search = DuckDuckGoSearchRun()

    try:
        # Attempt to use Tavily
        result = retriever.invoke(query)
        if result:
            return {"source": "Tavily", "results": result}
    except Exception as e:
        print(f"Tavily failed: {e}")

    try:
        # Fallback to DuckDuckGo
        result = duck_search.invoke(query)
        return {"source": "DuckDuckGo", "results": result}
    except Exception as e:
        print(f"DuckDuckGo failed: {e}")
        return {"source": None, "results": None}


# ---------------------- List Cleaner -------------------------------
def list_cleaner(input_data):
    """
    Clean input to ensure it is returned as a list.
    
    Args:
        input_data (str or list): Input data, either a list in string format or a direct list.
    
    Returns:
        list: The cleaned list.
    """
    if isinstance(input_data, list):
        # If the input is already a list, return it directly
        return input_data
    elif isinstance(input_data, str):
        try:
            # Attempt to parse the string as a Python literal
            parsed = ast.literal_eval(input_data)
            if isinstance(parsed, list):
                return parsed
            else:
                raise ValueError("Input string does not represent a list.")
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid input: {input_data}. Error: {e}")
    else:
        raise TypeError("Input must be a list or a string representing a list.")
    
# ------------------------------- Romanian Time ------------------------------------------------
def get_romania_time():
    """Get the current date and time in Romania."""
    from datetime import datetime
    from pytz import timezone
    
    romania_timezone = timezone('Europe/Bucharest')
    now = datetime.now(romania_timezone)
    return now.strftime('%Y-%m-%d %H:%M:%S')


# --------------------------------------------------- Chat Responses -----------------------------------------
def user_db_answer(user_query: str) -> str:
    logger.info("translating the user question")
    queries = multiquery_generator(user_query=user_query)
    cleaned_queries = list_cleaner(queries)
    logger.info("Searching the vec DB")
    rag_docs = most_similar_document(cleaned_queries)
    logger.info("Searching the internet")
    internet_search = internet_searcher(user_query)
    logger.info('Sending all info to AI to get the answer')
    answer = dbanswer_generator(user_query=user_query, internet_search=internet_search, db_result=rag_docs)
    return answer

def internet_answer(user_query: str) -> str:
    logger.info("Searching the internet")
    internet_search = internet_searcher(user_query)
    logger.info('Sending all info to AI to get the answer')
    answer = internet_search_ai(user_query=user_query, internet_search=internet_search)
    return answer

def direct_answer(user_query: str) -> str:
    logger.info('Generating direct answers')
    answer = direct_answer_ai(user_query=user_query)
    return answer

# --------------------------------------- CHATBOT ------------------------------------------
def chatbot(user_query:str) -> str:
    logger.info("Analyzing user query")
    analysis = query_analyzer(user_query=user_query)
    option = int(analysis)
    answer = None
    if option == 0: 
        answer = direct_answer(user_query=user_query)
    elif option == 1: 
        answer = internet_answer(user_query=user_query)
    elif option == 2: 
        answer = user_db_answer(user_query=user_query)
    return answer