import json
import logging
from pathlib import Path

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, ValidationError
from processor.import_processor.state import ImportGraphState


class NodeEntry(BaseNode):
    """
    入口节点：任务分发
    """

    name = "node_entry"

    def process(self, state: ImportGraphState):
        import_file_path = state.get("import_file_path")
        if not import_file_path:
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path",
                message="文件路径不能为空",
                expected_type=str)

        import_file_path_obj = Path("import_file_path")
        if import_file_path_obj.exists():
            raise FileProcessingError(
                message=f"文件{import_file_path_obj}不存在"
            )

        if import_file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif import_file_path_obj.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            raise ValidationError(
                message=f"文件{import_file_path_obj.suffix}格式不支持"
            )

        state["file_title"] = import_file_path_obj.stem

        return state



if __name__ == '__main__':

    setup_logging()

    init_state = {"import_file_path": "D:\\doc\\hak180产品安全手册.pdf"}

    node_entry = NodeEntry()
    result = node_entry(init_state)
    json_state = json.dumps(result, ensure_ascii= False, indent=4)

    logging.getLogger().info(json_state)