from abc import ABC, abstractmethod
from typing import TypeVar
from tool.logger import logger

from processor.query_processor.state import QueryGraphState
from tool.logger import logger

T = TypeVar("T")
class NodeBase(ABC):
    name: str = "base_node"

    def __call__(self, state: QueryGraphState) -> QueryGraphState:

        try:
            # 1. 开始准备执行节点
            logger.info(f"--- {self.name} 开始啦 ---")

            # 2. 执行节点
            result = self.process(state)

            # 3. 执行节点成功
            logger.info(f"--- {self.name} 完成啦 ---")

            return result

        except Exception as e:
            logger.error(f"{self.name} 执行失败: {e}")
            raise

    @abstractmethod
    def process(self, state: T) -> T:
        pass
