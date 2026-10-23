# RAG 知识库项目

基于 Python、LangChain 和 LangGraph 构建的 RAG（检索增强生成）学习与开发项目，面向产品说明书、操作手册等文档，探索从文档处理、向量入库到多路检索和问答的完整流程。

> 当前项目处于开发阶段，已包含文档处理、向量化、检索和重排序等模块代码。部分节点与配置仍需修正、联调，最终答案生成节点尚未实现，暂不作为开箱即用的完整问答系统。

## 项目目标

将产品文档转换为可检索的知识内容，并结合用户问题和历史对话，检索相关资料，为后续生成回答提供依据。

主要涉及以下能力：

- PDF 转 Markdown。
- Markdown 图片处理与图片内容理解。
- 文档切分与商品名称识别。
- BGE-M3 稠密向量和稀疏向量生成。
- Milvus 向量存储与混合检索。
- 结合历史对话进行商品名称确认和问题改写。
- 普通向量检索与 HyDE 检索。
- 通过 MCP 接入联网搜索。
- RRF 结果融合与 DashScope 文档重排序。
- MongoDB 会话记录存储。

以上为当前代码覆盖的模块范围，不代表完整流程已全部验证通过。

## 技术栈

| 技术 | 用途 |
| --- | --- |
| Python 3.11+ | 项目开发语言 |
| LangGraph | 编排文档导入与查询工作流 |
| LangChain / ChatOpenAI | 调用大语言模型 |
| BGE-M3 | 生成稠密向量与稀疏向量 |
| Milvus | 存储向量并执行混合检索 |
| MongoDB | 存储与读取会话记录 |
| MinIO | 存储文档中的图片资源 |
| MinerU API | 将 PDF 转换为 Markdown |
| DashScope | 模型调用与文本重排序 |
| MCP | 接入外部搜索工具 |
| uv | 管理 Python 依赖和虚拟环境 |

## 工作流程

### 文档导入流程

```text
PDF / Markdown 文档
        ↓
文件识别与任务分发
        ↓
PDF 转 Markdown（仅 PDF）
        ↓
图片处理与内容理解
        ↓
文档切分
        ↓
商品名称识别
        ↓
BGE-M3 向量化
        ↓
写入 Milvus
```

导入工作流入口：

```text
processor/import_processor/main_graph.py
```

### 查询流程

```text
用户问题 + 会话标识
        ↓
读取历史对话
        ↓
商品名称确认与问题改写
        ↓
        ├── 商品不明确：返回澄清提示
        │
        └── 商品已确认：进入多路检索
                    ↓
          ┌─────────┼─────────┐
          ↓         ↓         ↓
       向量检索   HyDE 检索   联网搜索
          └────┬────┘         │
               ↓              │
            RRF 融合          │
               └──────┬───────┘
                      ↓
                 文档重排序
                      ↓
              答案生成（待实现）
```

查询工作流入口：

```text
processor/query_processor/main_graph.py
```

其中，HyDE 检索先生成一段假设性回答，再将其用于知识库检索；该假设性回答本身不作为事实依据。

## 项目结构

```text
knowledge_base/
├── config/                       # 模型、向量库与 MCP 配置
├── processor/
│   ├── import_processor/         # 文档导入工作流
│   │   ├── nodes/                # 文档解析、切分、向量化等节点
│   │   ├── import_config.py      # 导入流程配置
│   │   ├── main_graph.py         # 导入流程编排
│   │   └── state.py              # 导入状态定义
│   └── query_processor/          # 查询工作流
│       ├── node/                 # 检索、融合、重排序等节点
│       ├── prompt/               # 提示词模板
│       ├── query/                # 商品名称确认相关提示词
│       ├── main_graph.py         # 查询流程编排
│       └── state.py              # 查询状态定义
├── utils/                        # 模型、数据库和存储工具
├── tool/                         # 日志、模型下载等辅助工具
├── test/                         # GPU 检测及模型测试脚本
├── .env.example                  # 环境变量配置示例
├── .gitignore                    # Git 忽略规则
├── pyproject.toml                # 项目信息与依赖声明
└── uv.lock                       # 依赖锁定文件
```

## 环境准备

### 1. 安装依赖

准备 Python 3.11 或更高版本，并安装 uv。在项目根目录执行：

```bash
uv sync
```

当前 `pyproject.toml` 为 PyTorch 配置了 CUDA 12.8 下载源。使用其他计算环境时，需要根据实际情况调整依赖源和模型设备配置。

### 2. 准备外部服务

根据需要运行的模块，准备以下服务：

| 服务 | 用途 |
| --- | --- |
| Milvus | 商品名称与文档切片的向量存储、检索 |
| MongoDB | 查询流程中的历史对话管理 |
| MinIO | 文档图片存储 |
| MinerU API | PDF 文档解析 |
| DashScope | 大模型调用及文本重排序 |
| Web Search MCP | 查询流程中的联网搜索 |

当前查询工作流连接了联网搜索分支，运行完整查询时需要配置对应服务。

### 3. 配置环境变量

复制 `.env.example` 为 `.env`。

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

Linux / macOS：

```bash
cp .env.example .env
```

如果本地已经存在 `.env`，直接编辑已有文件即可。

根据实际环境填写配置：

| 配置项 | 说明 |
| --- | --- |
| `OPENAI_API_KEY` | 模型服务 API 密钥 |
| `OPENAI_API_BASE` | OpenAI 兼容接口地址 |
| `LLM_DEFAULT_MODEL` | 默认语言模型 |
| `LLM_DEFAULT_TEMPERATURE` | 模型温度参数 |
| `VL_MODEL` | 图片理解模型 |
| `ITEM_MODEL` | 商品名称识别模型 |
| `BGE_M3_PATH` | 本地 BGE-M3 模型路径 |
| `BGE_DEVICE` | 模型运行设备 |
| `BGE_FP16` | 是否使用半精度 |
| `MILVUS_URL` | Milvus 服务地址 |
| `CHUNKS_COLLECTION` | 文档切片集合名称 |
| `ITEM_NAME_COLLECTION` | 商品名称集合名称 |
| `MONGO_URL` | MongoDB 连接地址 |
| `MONGO_DB_NAME` | MongoDB 数据库名称 |
| `MINIO_ENDPOINT` | MinIO 服务地址 |
| `MINIO_ACCESS_KEY` | MinIO 访问密钥 |
| `MINIO_SECRET_KEY` | MinIO 私有密钥 |
| `MINIO_BUCKET_NAME` | 图片存储桶名称 |
| `MCP_DASHSCOPE_BASE_URL` | 联网搜索 MCP 服务地址 |

当前 `.env.example` 尚未覆盖全部代码配置项，使用相关模块时还需要补充：

```dotenv
# PDF 解析服务
MINERU_API_TOKEN=your_mineru_token
MINERU_BASE_URL=your_mineru_api_base_url

# DashScope 文本重排序
TEXT_RERANK_MODEL=your_rerank_model
TEXT_RERANK_INSTRUCT=your_rerank_instruction

# 按图片存储需求设置
MINIO_IMG_DIR=images
```

模型名称、接口地址及本地路径应以实际使用的服务为准。MCP 服务地址需要与代码采用的传输协议匹配。

`.env` 用于保存真实配置，已被 `.gitignore` 忽略；提交代码时保留不含真实密钥的 `.env.example`。

### 4. 准备 BGE-M3 模型

项目提供模型下载脚本：

```text
tool/download_bgem3.py
```

先根据本机环境修改脚本中的缓存路径，再在项目根目录执行：

```bash
uv run python -m tool.download_bgem3
```

下载完成后，将实际模型目录填写到 `.env` 的 `BGE_M3_PATH` 中。

## 工作流调用示例

以下示例用于说明工作流接口。运行前需要完成环境配置，并处理“当前开发状态”中列出的待完善事项。

### 文档导入

```python
from processor.import_processor.main_graph import KBImportWorkflow

initial_state = {
    "import_file_path": r"D:\docs\产品手册.pdf",
    "file_dir": r"D:\output",
}

workflow = KBImportWorkflow()
result = workflow.run(initial_state)

print(result)
```

将文档路径和输出目录替换为自己的实际路径。

### 知识库查询

```python
from processor.query_processor.main_graph import KBQueryWorkflow

initial_state = {
    "session_id": "demo_session_001",
    "original_query": "请问这款产品如何调节温度？",
}

workflow = KBQueryWorkflow()
result = workflow.run(initial_state)

print(result.get("answer"))
print(result.get("reranked_docs"))
```

查询入口需要提供 `session_id` 和 `original_query`。连续对话应复用同一个 `session_id`，并尽量在问题中明确知识库已有的产品名称或型号。

最终答案生成节点尚未实现，因此当前应重点检查商品确认结果及检索、重排序的中间数据。

## 当前开发状态

项目已搭建导入和查询工作流，并包含主要处理节点的实现，仍有以下事项需要完善：

- 修正文档入口节点中的文件路径构造与存在性判断。
- 统一查询流程中的状态字段及配置属性名称。
- 更新主程序示例，使其与实际必填参数保持一致。
- 补全 `.env.example` 中缺失的配置项。
- 实现最终答案生成节点。
- 验证 MCP 接口地址与传输协议的兼容性。
- 完成外部服务联调及端到端测试。
- 后续补充 HTTP 接口和交互界面。

当前以 Python 工作流和节点脚本为主要使用方式。

## 开发注意事项

- 在项目根目录运行代码，确保包导入路径正确。
- 查询前需要准备对应的商品名称和文档切片数据。
- 模型路径、样例文档路径和服务地址需要按本机环境修改。
- MongoDB 工具模块会在导入时初始化连接，运行查询前应确保服务可用。
- 向量库集合的向量维度需要与实际嵌入模型输出一致。
- `.venv/`、缓存、日志、临时文件和本地 IDE 配置已通过 `.gitignore` 排除。
