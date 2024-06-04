import sys
import time
import torch
import numpy
from flask import Flask, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import chroma
from langchain.llms import llamacpp
from langchain.callbacks.manager import CallbackManager
from dotenv import load_dotenv
from src.prompt import *
from logs.logger import get_logger
from src.llm import LLMOutHandler

# Initialize logger
logger = get_logger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Define constants
index_name = "chat"

# Download and initialize embeddings
embeddings = download_hugging_face_embeddings()

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize LLM Output Handler
llm_custom = LLMOutHandler(device)

# Initialize document search with Chroma
docsearch = chroma.Chroma(persist_directory=index_name, embedding_function=embeddings)

# Initialize prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"]) 

# Define chain type keyword arguments
chain_type_kwargs = {"prompt": PROMPT}

# Initialize LlamaCpp model
llm = llamacpp.LlamaCpp(
    model_path=r"model/llama-2-7b-chat.Q5_K_M.gguf.bin",
    n_ctx=2048, 
    n_gpu_layers=100,
    n_batch=512, 
    n_threads=1, 
    model_kwargs={
        'model_type': 'llama', 
        'max_new_tokens': 1020, 
        'context_length': 1020, 
        'gpu_layers': 50
    },
    temperature=0.1,
    callback_manager=CallbackManager([llm_custom]),
    verbose=True
)

# Initialize RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 5}),
    return_source_documents=False, 
    chain_type_kwargs=chain_type_kwargs,
    verbose=True
)

@app.route("/")
def index():
    """
    Root URL route, simply returns a welcome message.
    """
    return jsonify({"message": "Welcome to the LLM API"})

@app.route("/get", methods=["POST"])
def chat():
    """
    API endpoint that handles the chat functionality based on user input.
    """
    start = time.time()
    data = request.get_json()
    input_msg = data.get("msg")
    if not input_msg:
        return jsonify({"error": "No input message provided"}), 400

    logger.info(f"Received input: {input_msg}")
    result = qa({"query": input_msg})
    response_msg = result["result"]
    logger.info(f"Response: {response_msg}")
    logger.info(f'Time taken: {round(time.time() - start, 2)} seconds')
    return jsonify({"response": response_msg})

if __name__ == '__main__':
    print('jmd shree ganesha')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    print("starting server")
    llm_custom = LLMOutHandler(device)
    app.run(host="0.0.0.0", port=8080, debug=True)