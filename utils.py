import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
import re
from dotenv import load_dotenv
from tavily import TavilyClient
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_core.pydantic_v1 import BaseModel
import time
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

memory = SqliteSaver.from_conn_string(":memory:")
embeddings = OpenAIEmbeddings()


_ = load_dotenv()


os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv('GROQ_API_KEY')
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

model_0 = ChatGroq(groq_api_key=groq_api_key,
             model_name="llama3-groq-70b-8192-tool-use-preview", temperature=0.8)
model_1 = ChatGroq(groq_api_key=groq_api_key,
             model_name="llama3-groq-8b-8192-tool-use-preview", temperature=0.3)
model_2 = ChatGroq(groq_api_key=groq_api_key, 
                model_name="llama3-70b-8192", temperature=0.2)
model_3 = ChatGroq(groq_api_key=groq_api_key,
                model_name="llama3-8b-8192", temperature=0.2)


# creeating a function to load the vector store
def load_vector_store():
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# a text cleaning function
def remove_symbols(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return cleaned_text

# Loading the vector store
vec_db = load_vector_store()

# document retriever function
def most_similar_document(query: list) -> list:
    all_docs = []
    query = list(set(query))
    for doc in query:
        query_docs = vec_db.similarity_search(doc)
        docs = [remove_symbols(i.page_content) for i in query_docs[:1]]
        all_docs.extend(docs)
    return all_docs

# Agent state
class ChatAgentState(TypedDict):
    question: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int
    
def doc_organizer(question: str, docs: list) -> str:
    
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert at Answering Questions.
    Background: The user is interested in knowing some information about the company Hidroelectrica. \n
    The user loves the company so much and wants to know more about it. \n


    The user asked the questions in USER_QUESTION. \n
    The database content is in Romanian. \n The retrieved documents are in RETRIEVED_DOCUMENTS. \n
    
    The retrieved documents are in Romanian. \n
    
    Here's your task: \n
        1. Translate the documents to English. \n
        2. Filter the translated document to keep only those related to the user \n
    
    As output return only parts that are related the question in one paragraph. \n
        
    Please only return the needed paragraph. No extra explanation.. Just that paragraph without any explanation. \n
    Your output should be in English. \n
    
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    USER_QUESTION: {question}
    RETRIEVED_DOCUMENTS: {docs}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    AI Assistant:"""

    question_prompt = PromptTemplate(input_variables=["question", "docs"], template=template)
    initiator_router = question_prompt | model_0 | StrOutputParser()
    output = initiator_router.invoke({"question":question, "docs":docs})
    return output

def multi_query_generator(question: str) -> str:
    
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert at Reviewing Questions.
    Background: The user is interested in knowing some information about the company Hidroelectrica. \n
    The client loves the company so much and wants to know more about it. \n


    The client will send in a question, your job is to review the question and create text sub-questions. \n The sub-questions are -questions written in a better and critical way to get more answers. 
    The database content is in  Romanian. Your job is to create two sub-questions in romanian \n
    Make sure the sub-questions are relevant to the client's question. \n

    Output format : [sub-questions1, sub-questions2] \n
    Return only the output without any additional information. \n
    
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    USER_QUESTION: {question}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    AI Assistant:"""

    question_prompt = PromptTemplate(input_variables=["question"], template=template)
    initiator_router = question_prompt | model_2 | StrOutputParser()
    output = initiator_router.invoke({"question":question})
    return output

# Combining the two functions
def rag_agent(question: str) -> str:
    # generating sub-questions
    questions = multi_query_generator(question)
    # retrieving documents
    docs = most_similar_document(query=questions)
    # generating response
    response = doc_organizer(question, docs)
    return response

# Planning Prompt
PLAN_PROMPT = """You are an chat expert good with creating a high level chat conversation with a user. You are tasked with writing and crafting a high level outline on how to answer a user question. \
Background: The user is interested in knowing some information about the company Hidroelectrica. \
The user loves the company so much and wants to know more about it. \
    
You observe the user question, then create an outline on the best way of answering the question \
This outline will serve as a plan on how to answer the question correctly. \
Give an outline of the answer along with any relevant notes or instructions on the answer tone and uniqueness.\
Please make it short and concise. \
"""

# Writer Prompt
WRITER_PROMPT = """You are a chat ai agent tasked with review chat messages response to the user.\
Generate the best answer possible for the user's question and the initial response. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

# Reflection Prompt
REFLECTION_PROMPT = """You are a teacher grading a chat response. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for tone, length, depth, style, etc."""


# research plan prompt
RESEARCH_PLAN_PROMPT = """You are a chat researcher charged with providing information that can \
be used when composing the a better chat response. Generate a list of search queries that will gather \
any relevant information. Only generate 2 queries max. it should be relevant to the response and the user's question.\
If there's no need for a query, respond with "None".    
"""

# research critique prompt
RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""

############ AGENT NODES ################
class Queries(BaseModel):
    queries: List[str]

# Plan Node    
def plan_node(state: ChatAgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT), 
        HumanMessage(content=state['question'])
    ]
    response = model_0.invoke(messages)
    return {"plan": response.content}

# Research Plan Node
def research_plan_node(state: ChatAgentState):
    
    # getting the rag document
    rag_doc = rag_agent(state['question'])
    
    content = state['content'] or []
    # adding the rag document to the content
    content.append(rag_doc)
    
    # creating the queries
    queries = model_3.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['question'])
    ])
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

# Generation Node
def generation_node(state: ChatAgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['question']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
        ]
    response = model_0.invoke(messages)
    return {
        "draft": response.content, 
        "revision_number": state.get("revision_number", 1) + 1
    }

# Reflection Node
def reflection_node(state: ChatAgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT), 
        HumanMessage(content=state['draft'])
    ]
    response = model_2.invoke(messages)
    return {"critique": response.content}


# Research Critique Node
def research_critique_node(state: ChatAgentState):
    queries = model_1.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}


# Should Continue Node
def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


##################### BUILDING THE GRAPH ############################

# Initialize the graph
builder = StateGraph(ChatAgentState)

# Setting the graph nodes
builder.add_node("planner", plan_node)
# builder.add_node("rag", rag_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)

# Starting to build the graph
# setting the entry point
builder.set_entry_point("planner")


# setting the conditional edge
builder.add_conditional_edges(
    "generate", 
    should_continue, 
    {END: END, "reflect": "reflect"}
)

# Connecting the remaining nodes
# rag -> research_plan -> generate -> reflect -> research_critique -> generate
builder.add_edge("planner", "research_plan")
# research_plan -> generate 
builder.add_edge("research_plan", "generate")
# generate --> Conditional node  --> reflect -> research_critique
builder.add_edge("reflect", "research_critique")
# research_critique -> generate
builder.add_edge("research_critique", "generate")


graph = builder.compile(checkpointer=memory)

def run_graph(question: str, max_revisions: int = 2, thread: dict = {}):
    for s in graph.stream({
    'question': question,
    "max_revisions": 2,
    "revision_number": 1,
    }, thread):
        print(s)
        
    # return the final state
    return s['generate']['draft']









