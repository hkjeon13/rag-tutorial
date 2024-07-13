from pydantic import BaseModel
from typing import List, Dict, Any


class Identification(BaseModel):
    id: str
    name: str
    group_id: str


class Document(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float = 0.0


class IndexingItem(Identification):
    documents: List[Document] = []
    max_chunk_size: int = 1024
    num_chunk_overlap: int = 256


class IndexingOutput(Identification):
    is_success: bool


class RetrievalItem(Identification):
    query: str
    max_query_size: int = 1024


class RetrievalOutput(Identification):
    related_documents: List[Document] = []


class Utterance(Identification):
    role: str
    content: str


class ChatItem(Identification):
    messages: List[Utterance] = []
    max_query_size: int = 1024
    max_response_size: int = 4096

# ChatOutput is not defined here, but it is defined in the API server code directly.
