# RAG Chatbot Prototype

This README document provides an overview of the RAG (Retrieval-Augmented Generation) Chatbot Prototype. The chatbot utilizes language models and vector database searches to generate contextually relevant responses based on input queries.

## Features

- Interaction with a chatbot through a web-based interface.
- Use of OpenAI's GPT-4 model for natural language understanding and response generation.
- Contextual answers based on a similarity search within the provided PDF documents.
- Integration of Gradio for creating a user-friendly web interface.
- Ability to add more documents to the vector database to enhance retrieval capability.

## Requirements

To run this chatbot, you will need the following:

- Python 3.8 or higher
- An OpenAI API key (GPT-4 access)

## Setup

### Install Dependencies

First, ensure that you have all required packages installed by running:

```bash
pip install gradio langchain langchain-community langchain-openai dotenv
```

### Environment Variables

Create a `.env` file in the root directory of your project and add your OpenAI API key:

```plaintext
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

To use the chatbot, follow these steps:

1. Place your PDF files in the `./pdf/` directory.
2. Run the script with:

   ```bash
   python app.py
   ```

3. The chatbot interface will be served on a local web server, and optionally, a public link is created if `share=True` is passed to `demo.launch()`.

## How It Works

The system performs the following operations:

- Loads environment variables using `dotenv`.
- Splits text from PDF documents into chunks using `CharacterTextSplitter`.
- Creates vector embeddings for the text chunks with `OpenAIEmbeddings` and stores them in a chroma vector database (`Chroma`).
- When a query is received, it performs a similarity search against the vector database.
- A question-answering chain (`StuffDocumentsChain`) uses the retrieved documents to generate a response based on the context and the current user query.
- The response is then sent back to the Gradio web interface.

## Clearing Chat

A button is provided in the web interface to clear the chat history when needed.

## Development Notes

- Ensure that the `create_vectardb` function correctly initializes the `vectorstore` before attempting to add documents.
- Confirm that the `load_qa_chain` function's signature accords with the library's expected parameters.
- Debug mode is enabled in the Gradio interface launch for easier debugging during development.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
