import torch
from langchain.callbacks.base import BaseCallbackHandler


device = "cuda" if torch.cuda.is_available() else "cpu"
class LLMOutHandler(BaseCallbackHandler):
    def __init__(self, device):
        self.tokenstring = ''
        self.device = device
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokenstring += token