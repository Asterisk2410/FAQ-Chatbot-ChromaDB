from langchain.vectorstores import chroma
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from dotenv.main import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
dataset = r"C:\Users\Admin\Desktop\Aster\Chatbot-Llama2\data"

extracted_data = load_pdf(dataset)
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
index_name = "chat"

vectordb = chroma.Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory=index_name)

vectordb.persist()

