import sys , time, torch, numpy

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.llm import LLMChain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import llamacpp
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter


device = "cuda" if torch.cuda.is_available() else "cpu"
class LLMOutHandler(BaseCallbackHandler):
    def __init__(self, device):
        self.tokenstring = ''
        self.device = device
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokenstring += token