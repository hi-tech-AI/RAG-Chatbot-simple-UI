import os
import httpx
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex

from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
import re
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding


from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

import gradio as gr
import uuid

api_key = os.environ.get("API_KEY")
base_url = os.environ.get("BASE_URL")

llm = Cohere(
    api_key=api_key, 
    model="command")
embedding_model = CohereEmbedding(
    api_key=api_key, 
    model_name="embed-multilingual-v3.0",
    input_type="search_document",
    embedding_type="int8",)


memory = ""

# Set Global settings
Settings.llm = llm
Settings.embed_model=embedding_model
# set context window
Settings.context_window = 4096
# set number of output tokens
Settings.num_output = 512



db_path=""

def validate_url(url):
    try:
        response = httpx.get(url, timeout=60.0)
        response.raise_for_status()
        text = [Document(text=response.text)]
        option = "web"
        return text, option
    except httpx.RequestError as e:
        raise gr.Error(f"An error occurred while requesting {url}: {str(e)}")
    except httpx.HTTPStatusError as e:
        raise gr.Error(f"Error response {e.response.status_code} while requesting {url}")
    except Exception as e:
        raise gr.Error(f"An unexpected error occurred: {str(e)}")

def extract_web(url):
    print("Entered Webpage Extraction")
    prefix_url = "https://r.jina.ai/"
    full_url = prefix_url + url
    print(full_url)
    print("Exited Webpage Extraction")
    return validate_url(full_url)

def extract_doc(path):
    documents = SimpleDirectoryReader(input_files=path).load_data()
    option = "doc"
    return documents, option


def create_col(documents):
    # Create a client and a new collection
    db_path = f'database/{str(uuid.uuid4())[:4]}'
    client = chromadb.PersistentClient(path=db_path)
    chroma_collection = client.get_or_create_collection("quickstart")
    
    # Create a vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Create a storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Create an index from the documents and save it to the disk.
    VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    return db_path

def infer(message:str, history: list):
    global db_path
    global memory
    option=""
    print(f'message: {message}')
    print(f'history: {history}')
    messages = []
    files_list = message["files"]
    
         
    if files_list:
        documents, option = extract_doc(files_list)
        db_path = create_col(documents)
        memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    else:
        if message["text"].startswith("http://") or message["text"].startswith("https://"):
            documents, option = extract_web(message["text"])
            db_path = create_col(documents)
            memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
        elif not message["text"].startswith("http://") and not message["text"].startswith("https://") and len(history) == 0:
            raise gr.Error("Please send an URL or document")
            

    # Load from disk
    load_client = chromadb.PersistentClient(path=db_path)
    
    # Fetch the collection
    chroma_collection = load_client.get_collection("quickstart")
    
    # Fetch the vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Get the index from the vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store,
    )

    if option == "web" and len(history) == 0:
        response = "Getcha! Now ask your question."   
    else: 
        question = message['text']

        chat_engine = CondensePlusContextChatEngine.from_defaults(
            index.as_retriever(),
            memory=memory,
            context_prompt=(
                "You are an assistant for question-answering tasks."
                "Use the following context to answer the question:\n"
                "{context_str}"
                "\nIf you don't know the answer, just say that you don't know."
                "Use five sentences maximum and keep the answer concise."
                "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
            ),
            verbose=True,
        )
        response = chat_engine.chat(
            question
        )
        
    print(type(response))
    print(f'response: {response}')
    

    return str(response)
    


css="""
footer {
    display:none !important
}
h1 {
    text-align: center;
    display: block;
}
"""

title="""
<h1>RAG demo</h1>
<p style="text-align: center">Retrieval for web and documents</p>
"""


chatbot = gr.Chatbot(placeholder="Please send an URL or document file at first<br>Then ask question and get an answer.", height=800)

with gr.Blocks(theme="soft", css=css, fill_height="true") as demo:
    gr.ChatInterface(
        fn = infer,
        title = title,
        multimodal = True,
        chatbot = chatbot,
    )

if __name__ == "__main__":
    demo.queue(api_open=False).launch(show_api=False, share=False)