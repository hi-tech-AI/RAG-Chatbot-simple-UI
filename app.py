from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4-1106-preview", api_key=OPENAI_API_KEY)

vectordb_path = './vector_db'
uploaded_files = ['./pdf/airbus.pdf', './pdf/annualreport2223.pdf']
vectorstore = None

def create_vectordb():
    for file in uploaded_files:
        loader = PyPDFLoader(file)
        data = loader.load()
        texts = text_splitter.split_documents(data)

        if vectorstore is None:
            vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=os.path.join(vectordb_path))
        else:
            vectorstore.add_documents(texts)

def rag_bot(query, chat_history):
    print(f"Received query: {query}")

    template = """Please answer to human's input based on context. If the input is not mentioned in context, output something like 'I don't know'.
    Context: {context}
    Human: {human_input}
    Your Response as Chatbot:"""

    prompt_s = PromptTemplate(
        input_variables=["human_input", "context"],
        template=template
    )

    # Initialize vector store
    vectorstore = Chroma(persist_directory=os.path.join(vectordb_path), embedding_function=embeddings)

    docs = vectorstore.similarity_search(query)

    stuff_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_s)

    output = stuff_chain({"input_documents": docs, "human_input": query}, return_only_outputs=False)

    final_answer = output["output_text"]
    print(f"Final Answer ---> {final_answer}")

    return final_answer

def chat(query, chat_history):
    response = rag_bot(query, chat_history)
    # chat_history.append((query, response))
    return response

chatbot = gr.Chatbot(avatar_images=["user.jpg", "bot.png"], height=600)
clear_but = gr.Button(value="Clear Chat")
demo = gr.ChatInterface(fn=chat, title="RAG Chatbot Prototype", multimodal=False, retry_btn=None, undo_btn=None, clear_btn=clear_but, chatbot=chatbot)

if __name__ == '__main__':
    create_vectordb()
    demo.launch(debug=True, share=True)