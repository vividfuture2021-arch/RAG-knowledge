import json
from typing import Dict, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from config.llm_config import llm_config
from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase, T
from processor.query_processor.query.item_name_confirm import ITEM_NAME_EXTRACT_TEMPLATE, \
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from utils.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names


class NodeItemNameConfirm(NodeBase):
    name: str = "node_item_name_confirm"

    def process(self, state: T) -> T:
        """
               必要参数：session_id、original_query
               更新参数：history、rewritten_query、item_names、answer

               :param state: 工作流状态对象
               :return: 更新后的状态对象
               """

        # 步骤1：校验参数
        session_id, original_query = self._step_1_validate_param(state)
        logger.info(f"步骤1：参数校验通过")

        # 步骤2：获取历史记录
        history = get_recent_messages(session_id)
        logger.info(f"步骤2：获取到 {len(history)} 条历史消息")
        # 更新状态
        state["history"] = history

        # 步骤3：用户初始消息保存
        message_id = save_chat_message(session_id, "user", original_query)
        logger.info(f"步骤3：用户消息已初始保存, ID: {message_id}")

        # 步骤4：提取信息
        extract_res = self._step_4_extract_info(original_query, history)
        item_names = extract_res.get("item_names", [])
        rewritten_query = extract_res.get("rewritten_query", original_query)
        # 更新状态
        state["rewritten_query"] = rewritten_query
        state["item_names"] = item_names

        # 5. & 6. 如果有提取到商品名，进行搜索和对齐
        align_result = {}
        if len(item_names) > 0:
            query_results = self._step_5_vectorize_and_query(item_names)
            align_result = self._step_6_align_item_names(query_results)
        else:
            logger.info("Node: 未提取到商品名，跳过向量检索")

        # 7. 检查确认状态
        state = self._step_7_check_confirmation(state, align_result, history)

        # # 8. 写入最终历史
        self._step_8_write_history(state, session_id, rewritten_query, message_id)
        return state

    def _step_1_validate_param(self, state):
        """
        参数校验
        :param state: 工作流状态对象
        :return: session_id, original_query
        """
        original_query = state.get("original_query")
        if original_query is None:
            raise ValueError("参数 original_query 不能为空")

        session_id = state.get("session_id")
        if session_id is None:
            raise ValueError("参数 session_id 不能为空")
        return session_id, original_query

    def _step_4_extract_info(self, query, history) -> Dict:
        """
        利用LLM从当前问题以及历史会话中提取出主要询问的商品名称item_names（可多个，JSON列表形式）
        若商品名不够明确则返回空列表，同时根据上下文重新改写问题，保证问题独立完整
        :param query: 字符串 - 用户当前原始查询问题（如："这个多少钱？"）
        :param history: 列表[字典] - 近期会话历史，每条消息含role/text等字段，
                        格式：[{"role": "user/assistant", "text": "消息内容", "_id": "消息ID"}, ...]
        :return: 字典 - 提取结果，固定包含2个字段，格式：
                 {
                     "item_names": ["商品名1", "商品名2", ...],  # 提取的商品名列表，无则空列表
                     "rewritten_query": "改写后的完整问题"       # 包含商品名的独立问题，无则返回原始query
                 }
        """
        try:
            # 1. 获取llm客户端
            chat_model = ChatOpenAI(
                model=llm_config.item_model,
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
                temperature=llm_config.llm_temperature,
                # 开启JSON标准输出模式，强制模型返回可解析的json_object
                model_kwargs={
                    "response_format": {"type": "json_object"}
                }
            )

            # 2. 构造历史对话文本，拼接为"角色: 内容"的格式，供LLM做上下文理解
            history_text = ""
            for msg in history:
                role = msg.get("role")
                content = msg.get("text")
                history_text += f"{role}: {content}\n"

            # 3. 处理和动态拼接提示词
            # 为了把大括号当作 “普通字符” 保留下来，用双大括号 {{ 表示普通的左大括号 {，双大括号 }} 表示普通的右大括号 }。
            user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
                history_text=history_text,
                query=query
            )

            # 4. 构造LLM调用的消息列表，包含系统角色（定义助手身份）和用户角色（传入提示词）
            messages = [
                SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]

            # 5. 调用LLM客户端，发起请求获取结果
            response = chat_model.invoke(messages)
            content = response.content

            # 6. 数据清洗：处理LLM可能返回的代码块格式（如```json ... ```），去除包裹符
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")

            # 7. 数据解析：将JSON字符串转为字典
            result = json.loads(content)

            # 8. 健壮性处理：确保字段存在
            # 确保返回结果包含item_names字段，无则设为空列表
            if "item_names" not in result:
                result["item_names"] = []
            # 确保返回结果包含rewritten_query字段，无则复用原始查询
            if "rewritten_query" not in result:
                result["rewritten_query"] = query

            # 9. 给item_names 去除空格
            result["item_names"] = [
                name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
                for name in result["item_names"]
            ]

            # 10、返回解析后的提取结果
            return result

        except Exception as e:
            # 捕获所有异常（如LLM调用失败、JSON解析失败等），记录错误日志
            logger.error(f"大模型调用异常：{e}")
            # 异常时返回默认结果：空商品名列表+原始查询
            return {"item_names": [], "rewritten_query": query}

    def _step_5_vectorize_and_query(self, item_names) -> List[Dict]:
        """
           把分析出的item_names逐个向量化（BGEM3模型），并在Milvus向量数据库(kb_item_names)中执行混合搜索，获取匹配评分
           :param item_names: 列表[字符串] - 步骤4中 提取的商品名列表（如["苹果15", "华为P60"]）
           :return: 列表[字典] - 格式：
                [
                    {
                        "extracted_name": "提取的原始商品名",  # 如"苹果15"
                        "matches": [                          # 该商品名的TopN匹配结果，无则空列表
                            {
                                "item_name": "数据库中的商品名",  # Milvus中存储的标准化商品名
                                "score": 0.98                  # 混合搜索的相似度评分（0-1，越高越相似）
                            },
                            ...
                        ]
                    },
                    ...
                ]
        """



        embeddings = generate_embeddings(item_names)

        milvus_client = get_milvus_client()
        results = []

        if not milvus_client:
            logger.warning(f"Milvus 获取失败，跳过向量化处理")
            return results

        collections = milvus_config.item_name_collections

        try:
            for i in range(len(item_names)):
                dense_vector = embeddings.get("dense")[i]
                sparse_vector = embeddings.get("sparse")[i]

                reqs = create_hybrid_search_requests(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector
                )
                res = hybrid_search(
                    client=milvus_client,
                    collection_name=collections,
                    reqs=reqs,
                    output_fields=["item_name"]
                )
                results.append({
                    "extracted_name": item_names[i],
                    "matches": [
                        {
                            "item_name": item.get("entity").get("item_name"),
                            "score": item.get("score")
                        }
                        for item in res[0]
                    ]
                })
        except Exception as e:
            logger.warning(f"混合检索处理异常：{e}")

        return results

    def _step_6_align_item_names(self, query_results) -> dict:
        """
        6 根据Milvus搜索评分，逐个对齐step4提取的item_names，生成「确认商品名」和「候选商品名」
        对齐规则（优先级a>b>c>d）：
                a  如果只有一个匹配结果评分高于0.85 → 直接确认该商品名
                b  如果多条匹配结果评分超过0.85 → 优先取与原始提取名相同的，无则取分数最高的
                c  如果无0.85分以上结果 → 取分数≥0.6的最高前5个作为候选
                d  如果无0.6分及以上结果 → 不返回任何商品名（确认+候选均为空）
        :param query_results: 列表[字典] - step5的返回结果，每个商品名的搜索匹配数据（格式同step5返回值）
        :return: 字典 - 商品名对齐结果，包含确认列表和候选列表，格式：
                 {
                     "confirmed_item_names": ["确认商品名1", "确认商品名2"],  # 去重后的确认商品名，无则空列表
                     "options": ["候选商品名1", "候选商品名2", ...]          # 去重后的候选商品名，无则空列表
                 }
        """
        confirmed_item_names: List[str] = []

        options: List[str] = []

        for res in query_results:
            extracted_name = res.get("extracted_name")
            matches = res.get("matches")

            if not matches:
                continue


            high = [m for m in matches if m.get("score") > 0.85]
            mid = [m for m in matches if m.get("score") >= 0.65]

            if len(high) > 0:
                for m in high:
                    confirmed_item_names.append(m.get("item_name"))
                continue

            if len(mid) > 0:
                for m in mid[:3]:
                    options.append(m.get("item_name"))

        return {
            "confirmed_item_names": list(set(confirmed_item_names)),
            "options": list(set(options))
        }

    def _step_7_check_confirmation(self, state, align_result, history):
        """
        7 检查step6对齐后的商品名状态，分3种分支更新state，并同步更新历史消息的商品名关联
        :param state: 字典 - 原始会话状态，包含session_id/original_query等核心字段
        :param align_result: 字典 - step6的对齐结果
        :param history: 列表[字典] - 近期会话历史
        :return: 字典 - 更新后的会话状态，包含item_names/answer
        """
        confirm = align_result.get("confirmed_item_names", [])
        options = align_result.get("options", [])
        if confirm:
            ids_to_update = []
            for msg in history:
                if not msg.get("item_name"):
                    msg_id = msg.get("_id")
                    ids_to_update.append(str(msg_id))

            if ids_to_update:
                update_message_item_names(ids_to_update, confirm)
            state["item_names"] = confirm
            state["answer"] = ""
            return state

        if options:
            options_str = ",".join(options)
            state["answer"] = f"您是像问以下哪个产品：{options_str}?"
            state["item_names"] = []
            return state

        state["answer"] = "未找到相关产品，请提供准确的商品品牌、型号和描述。"
        return state

    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        """
         8 把本次处理的核心信息（用户问题、助手答案、商品名、改写查询）写入MongoDB的会话历史
         包含2个核心操作：1. 写入助手答案（若有）；2. 更新用户原始问题的关联信息
         :param state: 字典 - step6更新后的会话状态，包含answer/item_names等字段
         :param session_id: 字符串 - 会话唯一标识
         :param rewritten_query: 字符串 - step3改写后的完整问题
         :param message_id: 字符串 - 本次用户问题的消息唯一ID
         :return:
         """

        if state.get("answer"):
            save_chat_message(session_id=session_id,
                              role="assistant",
                              text= state.get("answer"),
                              item_names=state.get("item_names"))

            save_chat_message(
                session_id=session_id,
                role="user",
                text=state["original_query"],
                rewritten_query=rewritten_query,
                message_id=message_id,
                item_names=state.get("item_names")
            )


if __name__ == '__main__':

    init_state = {
        "session_id": "test_session_002",
        "original_query" : "怎么使用豆包"
    }
    node_item_name_confirm = NodeItemNameConfirm()
    result = node_item_name_confirm(init_state)
    print(result)