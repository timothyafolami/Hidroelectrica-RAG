import os
import uuid
from typing import List
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

#  ---------------- Local Imports ---------------------
from src.utils import get_romania_time, chatbot

load_dotenv()

os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


#  ----------------------------- Recall Memory -----------------------------
recall_vector_store = InMemoryVectorStore(OpenAIEmbeddings())



def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id


class KnowledgeTriple(TypedDict):
    subject: str
    predicate: str
    object_: str


@tool
def save_recall_memory(memories: List[KnowledgeTriple], config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    user_id = get_user_id(config)
    for memory in memories:
        serialized = " ".join(memory.values())
        document = Document(
            serialized,
            id=str(uuid.uuid4()),
            metadata={
                "user_id": user_id,
                **memory,
            },
        )
        recall_vector_store.add_documents([document])
    return memories


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    user_id = get_user_id(config)

    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id

    documents = recall_vector_store.similarity_search(
        query, k=3, filter=_filter_function
    )
    return [document.page_content for document in documents]

@tool
def get_current_time() -> str:
    """
    This tool is used to get the current time of the user timezone.
    """
    time = get_romania_time()
    return time


@tool
def chatbot_response(user_query:str) -> str:
    """summary

    Args:
        user_query (str): This is the user question. Anything outside the time question. 

    Returns:
        str: Returns an answer for the user based on the asked questions. 
    """
    response = chatbot(user_query=user_query)
    return response    