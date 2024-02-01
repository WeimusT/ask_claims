from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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

# Step 3: Load the document to a vector store
embeddings = OllamaEmbeddings() # Connect to local embedding
text_splitter = RecursiveCharacterTextSplitter() # Create a document splitter
documents = text_splitter.split_documents(docs) # Split document into chunks
vector = FAISS.from_documents(documents, embeddings) # Saving document chunks to vector database

# Step 4: Create a docment chain
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context
<context>
{context}
</context>

Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)

# Step 5: Create a retriver chain. For a given question we can use the retriever to dynamically select the most relevant documents and pass those in.
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Step 6: Ask the question with retrieval chain
response = retrieval_chain.invoke({"input": "what is a langchain chain?"})
print(response["answer"])




