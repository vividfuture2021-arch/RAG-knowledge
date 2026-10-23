from http import HTTPStatus
from typing import List

import dashscope

from config.reranker_config import reranker_config


def rerank_documents(query: str, documents: List[str]) -> list[float]:
    dashscope.api_key = reranker_config.text_rerank_api_key
    resp = dashscope.TextReRank.call(
        model=reranker_config.text_rerank_model,
        query=query,
        documents=documents,
        top_n=len(documents),
        return_documents=False,
        instruct=reranker_config.text_rerank_instruct
    )
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"DashScope API error: {resp.status_code}")

    results = resp.output.get("results",[])
    scores = [0.0] * len(documents)
    for result in results:
        score = result.get("relevance_score")
        index = result.get("index")
        scores[index] = score
    return scores
