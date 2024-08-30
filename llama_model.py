import os
import subprocess
import time
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.llms.ollama import Ollama

from zipfile import ZipFile

# Unzipping the data.zip file and extracting its contents into the 'data' directory
zip_path = 'data.zip'
with ZipFile(zip_path, 'r') as zip:
    zip.extractall('data')



# Load JSON files
base_dir = 'data'
loaders = {
    '.json': JSONLoader
}



# Function to create a DirectoryLoader for a specific file type
def create_directory_loader(file_type, directory_path):
    return DirectoryLoader(
        path=directory_path,
        glob=f'**/*{file_type}',
        loader_cls=loaders[file_type],
    )



# Initialize the JSON loader for the '.json' files in the base directory
json_loader = create_directory_loader('.json', base_dir)

# Add the JSON loader to the list of loaders
loaders = [json_loader]
docs = []

# Load all documents using the loaders
for loader in loaders:
    docs.extend(loader.load())

# Print the number of loaded documents
print(f'Loaded {len(docs)} documents')




# Initialize HuggingFace embeddings using the BGE (BERT-based General Embedding) model
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cuda"}  # Set device to GPU
encode_kwargs = {"normalize_embeddings": True}  # Normalize the embeddings
embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)




# Initialize FAISS vector store and add documents to it
db = None

for d in tqdm(docs):
    if db:
        db.add_documents([d])
    else:
        db = FAISS.from_documents([d], embeddings)

# Save the FAISS index locally
db.save_local('FAISS_index')



# Start the Ollama service in the background
ollama_process = subprocess.Popen(["ollama", "serve"], shell=True)
time.sleep(15)  # Wait for the service to start

# Pull the specific model 'llama3.1:8b' from the Ollama service
subprocess.run(["ollama", "pull", "llama3.1:8b"], shell=True)


# Initialize the Ollama model with the specified parameters
llm = Ollama(model="llama3.1:8b", temperature=0.2, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


# Create a Retrieval-based QA chain using the LLM and the FAISS retriever
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(search_kwargs={"k": 5}),  # Retrieve top 5 relevant documents
    return_source_documents=True  # Return source documents along with the answers
)



