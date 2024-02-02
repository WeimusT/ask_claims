'''A retrieval chain is a sequence of hardcoded actions.
'''
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

# Step 1: Creae a large language model
with open("res/key", "rb") as fi:
    param = pickle.load(fi)
llm = ChatOpenAI(model="gpt-3.5-turbo", **param)
llm.invoke("what is a langchain chain?") # Test connection
# There is no specific information available about a "langchain chain." It is possible that you may be referring to a term or concept that is not widely recognized or used. Can you provide more context or clarify your question?

# Step 2: Retrieve documents using a web document loader
loader = WebBaseLoader("https://python.langchain.com/docs/modules/chains")
docs = loader.load()

# Step 3: Load the document to a vector store and create a retriever
embeddings = OllamaEmbeddings() # Connect to local embedding
text_splitter = RecursiveCharacterTextSplitter() # Create a document splitter
documents = text_splitter.split_documents(docs) # Split document into chunks
vector = FAISS.from_documents(documents, embeddings) # Saving document chunks to vector database
retriever = vector.as_retriever()

# Step 4: Create a docment chain. The prompt template is created from a template string.
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context
<context>
{context}
</context>

Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)

# Step 5: Create a retriver chain. For a given question we can use the retriever 
# to dynamically select the most relevant documents and pass those in.
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "what is a langchain chain?"})
print(response["answer"])

# Step 6: Create a prompt template from messages history. Create a chain that takes conversation history and returns documents.
message_history_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_message_history_chain = create_stuff_documents_chain(llm, message_history_prompt)
retrieval_chain = create_retrieval_chain(retriever, document_message_history_chain)
chat_history = [
    HumanMessage(content="Can LangChain chain be used to call LLM?"),
    AIMessage(content="Yes!")
]
response = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
print(response["answer"])


