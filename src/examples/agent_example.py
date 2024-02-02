'''An agent lets LLM to decide what steps to take. The 
'''
import os
import pickle
from langchain_openai import ChatOpenAI # ChatGPT model API
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor

# Step 1: Create a large language model
with open("res/openai.key", "rb") as fi:
    param = pickle.load(fi)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, **param) # Temperature is set to 0 to get the most accurate answer
help(ChatOpenAI)
# llm.invoke("what is a langchain chain?") # Test connection
# There is no specific information available about a "langchain chain." It is possible that you may be referring to a term or concept that is not widely recognized or used. Can you provide more context or clarify your question?

# Step 2: Retrieve documents using a web document loader
loader = WebBaseLoader("https://python.langchain.com/docs/modules/chains")
docs = loader.load()

# Step 3: Load the document to a vector store and create a retriever
embeddings = OllamaEmbeddings() # Connect to local embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Create a document splitter
documents = text_splitter.split_documents(docs) # Split document into chunks
vector = FAISS.from_documents(documents, embeddings) # Saving document chunks to vector database
retriever = vector.as_retriever()
# retriever.get_relevant_documents("What is longchain chain?") # Test retriever. It should return the document that contains the answer to the question.

# Step 4: Create a retriver tool. The description of the tool is passed to the LLM to help it decide what tool to use.
# So the description should be clear and concise.
retriever_tool = create_retriever_tool(
    retriever,
    "langchain_search",
    "Search for information about LangChain. For any questions about LangChain, you must use this tool!"
)

# Step 5: Create a Tavily tool. The Tavily tool is used to search for information on the web.
with open("res/tavily.key", "rb") as fi:
    tavily_api_key = pickle.load(fi)
os.environ["TAVILY_API_KEY"] = tavily_api_key
search = TavilySearchResults() # Connect to Tavily API
# search.invoke("what is the weather in Dublin?") # Test connection

# Step 6: Load a public prompt from langchain hub
prompt = hub.pull("hwchase17/openai-functions-agent")
# prompt.messages

# Step 7: Initialize the agent with the language model, tools, and prompt.
# The agent takes in input from the user and decides what tool to use to answer the question.
tools = [search, retriever_tool] # create a list of tools
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Step 8: Test the agent
agent_executor.invoke({"input": "hi!"})
agent_executor.invoke({"input": "what is a LangChain chain?"}) # This query matches the description of the retriever tool, so the agent should use the retriever tool to answer the question.
agent_executor.invoke({"input": "what is the weather in Dublin?"}) # This query should use the Tavily tool to answer the question.