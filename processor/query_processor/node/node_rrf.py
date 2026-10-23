from typing import List, Tuple, Dict

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger


class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web）进行加权融合排序。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        logger.info(f"【{self.name}】节点逻辑")
        embedding_chunks_list = state.get("embedding_chunks") or []
        hyde_embedding_chunks_list = state.get("hyde_embedding_chunks") or []

        embedding_search_list = [doc.get("entity") for doc in embedding_chunks_list if isinstance(doc, dict)]
        hyde_embedding_search_list = [doc.get("entity") for doc in hyde_embedding_chunks_list if isinstance(doc, dict)]
        rrf_inputs = [
            (embedding_search_list, 1.0),
            (hyde_embedding_search_list, 1.0)
        ]

        rrf_merge_results = self._rrf_merge(rrf_inputs)
        rrf_chunks = [doc for doc, score in rrf_merge_results]

        state["rrf_chunks"] = rrf_chunks

        return state
    def _rrf_merge(self, rrf_inputs, k: int=60, max_result: int=None) -> List[Tuple[Dict, float]]:
        """
        RRF 融合
        :param rrf_inputs: 输入数据
        :param k: 输入数据中，每路数据取前 k 个
        :param max_result: 最终结果数量限制
        :return: 融合后的结果
        """
        chunk_data = {}
        chunk_score = {}
        for rrf_input, weight in rrf_inputs:
            for rank, doc in enumerate(rrf_input, start=1):
                chunk_id = doc.get("chunk_id")
                chunk_score[chunk_id] = chunk_score.get(chunk_id, 0.0) + weight/(k+rank)
                chunk_data.setdefault(
                    chunk_id,
                    doc
                )
        unsorted_results = [(chunk_data[chunk_id], score) for chunk_id, score in chunk_score.items()]
        sorted_results = sorted(unsorted_results, key=lambda x: x[1], reverse=True)
        return sorted_results[:max_result] if max_result else sorted_results



if __name__ == '__main__':

    # 模拟两路检索结果
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#3"}},
        ]
    }

    node_rrf = NodeRrf()
    result = node_rrf(mock_state)
  
