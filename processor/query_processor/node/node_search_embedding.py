from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase, T
from processor.query_processor.node.node_search_embedding_hyde import NodeSearchEmbeddingHyde
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search


class NodeSearchEmbedding(NodeBase):
     """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    """

     # 覆盖基类的 name 属性，标识节点名称
     name: str = "node_search_embedding"

     def process(self, state: T) -> T:
         """
         节点逻辑
         :param state: 工作流状态对象
         :return: 更新后的状态对象
         """
         try:
             # TODO
             query = state.get("rewriter_query")
             item_names = state.get("item_names")

             embeddings = generate_embeddings([query])
             dense_vector = embeddings.get("dense")[0]
             sparse_vector = embeddings.get("sparse")[0]
             expr = f"item_name in {item_names}"

             reqs = create_hybrid_search_requests(
                 dense_vector,
                 sparse_vector,
                 expr=expr,
                 limit=10
             )


             client = get_milvus_client()
             collection_name = milvus_config.chunks_collection
             search_res = hybrid_search(
                 client=client,
                 collection_name=collection_name,
                 reqs=reqs,
                 ranker_weights=(0.8,0.2),
                 output_fields=["chunk_id", "content", "item_name"]
             )



             # return state
             return {"embedding_chunks": search_res[0] if search_res else []}

         except Exception as e:
             logger.error(f"向量检索失败：{e}")
             return {}


if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": ["BrotherHAK180烫金机", "BrotherHAK-180烫金机"]
    }
    node_search_embedding_hyde = NodeSearchEmbeddingHyde()
    result = node_search_embedding_hyde(init_state)
