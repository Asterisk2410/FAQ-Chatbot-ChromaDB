from langchain.vectorstores import chroma
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from dotenv.main import load_dotenv
from logs.logger import get_logger
import os

load_dotenv()

logger = get_logger(__name__)

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
dataset = r"/FAQ Chatbot-ChromaDB/data"

extracted_data = load_pdf(dataset)
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
index_name = "chat"
directory = r"/FAQ Chatbot-ChromaDB/chat"

if not os.path.exists(directory):
    logger.info("Creating vector database")
    vectordb = chroma.Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory=index_name)
    vectordb.persist()
else:
    logger.info("vector database already exists")
    pass