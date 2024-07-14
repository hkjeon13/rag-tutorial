import asyncio
import logging
import re
from typing import Generator

import uvicorn
from fastapi import FastAPI
from transformers import HfArgumentParser

from utils import (
    IndexingItem,
    IndexingOutput,
    RetrievalItem,
    RetrievalOutput,
    ChatItem,
    StreamingResponse,
    FastApiArgument
)

logger = logging.getLogger(__name__)
parser = HfArgumentParser((FastApiArgument,))
(fastapi_args,) = parser.parse_args_into_dataclasses()
app = FastAPI()


@app.post("/indexing")
async def indexing(item: IndexingItem) -> IndexingOutput:
    """
    문서를 입력 받아 색인(indexing) 작업을 합니다.
    Args:
        item: IndexingItem: 색인 작업을 위한 입력 데이터
            - id: str: 색인 작업을 위한 ID
            - name: str: 색인 작업을 위한 이름
            - group_id: str: 색인 작업을 위한 그룹 ID
            - documents: List[Document]: 색인할 문서 목록
            - max_chunk_size: int: 색인할 문서의 최대 chunk size
            - num_chunk_overlap: int: 색인할 문서의 chunk 간 겹치는 길이
    Returns:
        IndexingOutput: 색인 작업 결과
            - id: str: 색인 작업을 위한 ID
            - name: str: 색인 작업을 위한 이름
            - group_id: str: 색인 작업을 위한 그룹 ID
            - is_success: bool: 색인 작업 성공 여부
    """
    # TODO: 색인 작업을 수행하는 코드를 작성합니다. 현재는 성공으로 가정합니다.

    return IndexingOutput(
        id=item.id,
        name=item.name,
        group_id=item.group_id,
        is_success=True
    )


@app.post("/retrieve")
async def retrieval(item: RetrievalItem) -> RetrievalOutput:
    """
    질의(query)를 입력 받아 검색(retrieval) 작업을 합니다.
    Args:
        item: RetrievalItem: 검색 작업을 위한 입력 데이터.
            - id: str: 검색 작업을 위한 ID.
            - name: str: 검색 작업을 위한 이름.
            - group_id: str: 검색 작업을 위한 그룹 ID.
            - query: str: 검색할 질의.
            - max_query_size: int: 검색할 질의의 최대 길이.
            - top_k: int: 검색 결과 중 상위 몇 개를 가져올지 결정. (default: 3)
    Returns:
        RetrievalOutput: 검색 작업 결과.
            - id: str: 검색 작업을 위한 ID.
            - name: str: 검색 작업을 위한 이름.
            - group_id: str: 검색 작업을 위한 그룹 ID.
            - related_documents: List[Document]: 검색 결과로 나온 문서 목록.
    """

    # TODO: 검색 작업을 수행하는 코드를 작성합니다. 현재는 고정된 결과로 가정합니다.

    related_documents = [
        {
            "id": "doc1",
            "text": "This is the first document.",
            "metadata": {},
            "score": 0.9
        },
        {
            "id": "doc2",
            "text": "This is the second document.",
            "metadata": {},
            "score": 0.8
        }
    ]

    return RetrievalOutput(
        id=item.id,
        name=item.name,
        group_id=item.group_id,
        related_documents=related_documents
    )


@app.post("/chat")
async def chat(item: ChatItem):
    """
    챗봇을 위한 대화(chat) 작업을 합니다.
    Args:
        item: ChatItem: 대화 작업을 위한 입력 데이터.
            - id: str: 대화 작업을 위한 ID.
            - name: str: 대화 작업을 위한 이름.
            - group_id: str: 대화 작업을 위한 그룹 ID.
            - messages: List[Utterance]: 대화할 메시지 목록.
            - max_query_size: int: 대화할 질의의 최대 길이.
            - max_response_size: int: 대화할 응답의 최대 길이.
            - top_k: int: 검색 결과 중 상위 몇 개를 가져올지 결정. (default: 3)
            - stream: bool: 대화 작업 결과를 스트리밍할지 여부. (default: False)
    Returns:
        StreamingResponse | str: 대화 작업 결과.
            - StreamingResponse: 대화 작업 결과를 스트리밍하는 경우.
            - str: 대화 작업 결과를 한 번에 반환하는 경우.
    """

    # 검색 작업을 수행합니다.

    query = item.messages[-1].content

    related_documents = await retrieval(
        RetrievalItem(
            id=item.id,
            name=item.name,
            group_id=item.group_id,
            query=query,
            max_query_size=item.max_query_size,
            top_k=item.top_k
        )
    )

    # 검색 결과와 대화 메시지를 결합합니다.
    if related_documents.related_documents:
        retrieval_str = "\n\n".join(doc.text for doc in related_documents.related_documents)
        query = "{context}\n\n{query}".format(context=retrieval_str, query=query)

    messages = item.messages[:-1] + [{"role": "user", "content": query}]

    logger.info(f"[TEST] Chat messages:\n\n{messages}")

    # 대화 작업을 수행합니다.
    # TODO: 대화 작업을 수행하는 코드를 작성합니다. 현재는 고정된 리스트 값으로 가정합니다.
    contents = "This is the first response." * 10
    contents = re.split("( )", contents)

    # item.stream 이 False 일 경우, 한 번에 반환.
    if not item.stream:
        return "".join(contents)

    # item.stream 이 True 일 경우, StreamingResponse로 반환.
    # 현재 코드에서는 genertor를 사용하여 bytes 형태로 반환(이후 생성 모델의 스트림 출력으로 대체).
    async def generate_response() -> Generator:
        for content in contents:
            yield content
            await asyncio.sleep(0.02)

    return StreamingResponse(
        generate_response(),
        model_type="Others",
        db_manager=None,
        metadata=None
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=fastapi_args.server_address, port=fastapi_args.server_port)
