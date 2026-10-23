from typing import List, Dict, Any

from modelscope.preprocessors.templates.utils import upper_bound

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.reranker_http_utils import rerank_documents


class NodeRerank(NodeBase):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rerank"

    # -----------------------------
    # Rerank / TopK 全局常量
    # -----------------------------
    # 动态 TopK 硬上限：最多取前 N 条（<=10）
    RERANK_MAX_TOPK: int = 10
    # 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
    RERANK_MIN_TOPK: int = 3  # 总数最少条数

    # 断崖阈值（绝对，判断高分文档）
    RERANK_GAP_ABS: float = 0.5
    # 断崖阈值（相对，判断低分文档）
    RERANK_GAP_RATIO: float = 0.25

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        """
                执行重排序
                流程: 合并多源文档 → Reranker 计算相关性 → 断崖检测动态截断
                :param state: 需包含 rrf_chunks、web_search_docs、rewritten_query
                :return: 更新后的 state，包含 reranked_docs
                """

        # 1. 合并多源文档
        merged_multi_docs: List[Dict[str, Any]] = self._step_1_merge_multi_source_docs(state)

        # 2. Rerank 精排(精排打分)
        reranked_docs: List[Dict[str, Any]] = self._step_2_rerank_merged_docs(state, merged_multi_docs)

        # 3. 动态 Top_K 截取(断崖检测)
        cutoff_docs = self._step_3_cliff_cutoff(reranked_docs)

        # 4. 更新state
        state['reranked_docs'] = cutoff_docs

        return state

    def _step_1_merge_multi_source_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """合并本地 RRF 结果和网络搜索结果为统一格式"""
        rrf_chunks = state.get("rrf_chunks")
        web_search_docs = state.get("web_search_docs")

        final_docs = []

        for rrf_chunk in rrf_chunks:
            final_rrf_doc = {
                "chunk_id": rrf_chunk.get("chunk_id"),
                "content": rrf_chunk.get("content"),
                "title": rrf_chunk.get("item_name"),
                "url": None,
                "source": "local"
            }
            final_docs.append(final_rrf_doc)
        for web_doc in web_search_docs:
            final_web_doc = {
                "chunk_id": None,
                "url": web_doc.get("url"),
                "title": web_doc.get("title"),
                "content": web_doc.get("snippet"),
                "source": "web"
            }
            final_docs.append(final_web_doc)
        return final_docs

    def _step_2_rerank_merged_docs(self, state: QueryGraphState, merged_multi_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用 Reranker 模型对文档进行精排"""
        user_query = state.get("rewritten_query")
        contents = [doc.get() for doc in merged_multi_docs]
        rerank_results = rerank_documents(user_query, contents)
        reranker_docs = [
            {
                "chunk_id": doc.get("chunk_id"),
                "url": doc.get("url"),
                "title": doc.get("title"),
                "content": doc.get("content"),
                "source": doc.get("source"),
                "score": score
            }
            for doc, score in zip(merged_multi_docs, rerank_results)
        ]
        sorted_docs = sorted(reranker_docs, key=lambda x: x["score"], reverse=True)
        return sorted_docs

    def _step_3_cliff_cutoff(self, ranked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """断崖检测截断：相邻得分差距超过阈值时截断。"""
        if not ranked_docs:
            return []
        upper_bound = min(self.RERANK_MAX_TOPK, len(ranked_docs))
        lower_bound = min(self.RERANK_MIN_TOPK, upper_bound)

        cutoff_pos = upper_bound
        for idx in range(lower_bound-1, upper_bound-1):
            current_score = ranked_docs[idx].get("score")
            next_score = ranked_docs[idx+1].get("score")
            if current_score is None or next_score is None:
                continue
            abs_gap = current_score - next_score
            rel_gap = abs_gap / (abs(current_score) + 1e-6)
            if abs_gap >= self.RERANK_GAP_ABS or rel_gap >= self.RERANK_GAP_RATIO:
                cutoff_pos = idx + 1
                break

        return ranked_docs[:cutoff_pos]


if __name__ == "__main__":

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {
                "chunk_id": "local_1",
                "item_name": "主板维修手册",
                "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"
            },
            {
                "chunk_id": "local_2",
                "item_name": "闲聊",
                "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"
            },
        ],
        "web_search_docs": [
            {
                "url": "https://example.com/repair",
                "title": "短路查修指南",
                "snippet": "主板通电前先打各主供电电感对地阻值，阻值偏低就是短路。"
            },
            {
                "url": "https://example.com/news",
                "title": "科技新闻",
                "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"
            },
        ],
    }

    node_rerank = NodeRerank()
    result = node_rerank(mock_state)