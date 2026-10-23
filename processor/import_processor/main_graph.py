from mimetypes import inited

from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.import_processor.nodes.node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.node_entry import NodeEntry
from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus
from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.nodes.node_md_img import NodeMDImg
from processor.import_processor.nodes.node_pdf_to_md import NodePDFToMD
from processor.import_processor.state import ImportGraphState


class KBImportWorkflow:

    def __init__(self, config=None):


        self._compiled_graph = None

    @property
    def graph(self):
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph()
        return self._compiled_graph

    @staticmethod
    def route_after_entry(state: ImportGraphState):
        if state.get("is_pdf_read_enabled"):
            return "node_pdf_to_md"
        elif state.get("is_md_read_enabled"):
            return "node_md_img"
        else:
            return END


    def build_graph(self):

         graph = StateGraph(ImportGraphState)


         graph.add_node("node_entry", NodeEntry())
         graph.add_node("node_pdf_to_md", NodePDFToMD())
         graph.add_node("node_md_img", NodeMDImg())
         graph.add_node("node_document_split", NodeDocumentSplit())
         graph.add_node("node_item_name_recognition", NodeItemNameRecognition())
         graph.add_node("node_bge_embedding", NodeBGEEmbedding())
         graph.add_node("node_import_milvus", NodeImportMilvus())

         graph.set_entry_point("node_entry")

         graph.add_conditional_edges(
             "node_entry",
             self.route_after_entry,
             {
                 "node_md_img": "node_md_img",
                 "node_pdf_to_md": "node_pdf_to_md",
                 END: END
             }
         )

         graph.add_edge("node_pdf_to_md", "node_md_img")
         graph.add_edge("node_md_img", "node_document_split")
         graph.add_edge("node_document_split", "node_item_name_recognition")
         graph.add_edge("node_item_name_recognition", "node_bge_embedding")
         graph.add_edge("node_bge_embedding", "node_import_milvus")
         graph.add_edge("node_import_milvus", END)

         return graph.compile()

    def run(self, state: ImportGraphState, stream: bool=False):

        self.graph.get_graph().print_ascii()

        if stream:
             return self.graph.stream(state, stream_mode="values")
        else:
            return self.graph.invoke(state)

if __name__ == '__main__':
    init_state = {
        "import_file_path": r"D:\doc\hak180产品安全手册.pdf",
        "file_dir": r"D:\output",
    }

    workflow = KBImportWorkflow()


    for event in workflow.run(init_state, stream=True):
        print(f"state: {event}")