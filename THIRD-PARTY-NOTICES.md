# 第三方组件与来源

课程来源及公开许可范围见 [COPYRIGHT.md](COPYRIGHT.md)。不删除课程原有署名；不将第三方框架的许可证套用为整库许可证。

本仓库通过 `pyproject.toml` / `uv.lock` 依赖下列组件，没有 vendor 或复制它们的实现。具体版本、构建平台与包摘要以锁文件为准；软件包中的 LICENSE / NOTICE 及其传递依赖条款继续适用。

| 组件 | 用途 | 上游许可入口 |
| --- | --- | --- |
| FastAPI | HTTP API | https://github.com/fastapi/fastapi/blob/master/LICENSE |
| Pydantic / pydantic-settings | 配置与数据契约 | https://github.com/pydantic/pydantic/blob/main/LICENSE |
| LangGraph | 状态图与checkpoint接口 | https://github.com/langchain-ai/langgraph/blob/main/LICENSE |
| LangChain / langchain-openai | 消息、模型与工具适配 | https://github.com/langchain-ai/langchain/blob/master/LICENSE |
| OpenAI Python SDK | OpenAI兼容协议客户端 | https://github.com/openai/openai-python/blob/main/LICENSE |
| MCP Python SDK | MCP客户端与本地服务 | https://github.com/modelcontextprotocol/python-sdk/blob/main/LICENSE |
| PyMilvus | Dense/BM25/RRF检索接口 | https://github.com/milvus-io/pymilvus/blob/master/LICENSE |
| SQLAlchemy | SQL会话与数据访问 | https://github.com/sqlalchemy/sqlalchemy/blob/main/LICENSE |
| HTTPX | HTTP与测试Transport | https://github.com/encode/httpx/blob/master/LICENSE.md |
| OpenEvals（可选） | 实验评审适配 | https://github.com/langchain-ai/openevals/blob/main/LICENSE |
| Langfuse（可选） | 原工程观测适配 | https://github.com/langfuse/langfuse/blob/main/LICENSE |

Docker Compose 只引用 MySQL、Milvus、etcd、MinIO 的上游镜像，不包含镜像内容。各镜像发行的许可（包括 MySQL、MinIO）独立适用，不能把Python库的MIT/Apache许可扩展到镜像。此清单不是软件物料清单或对全部传递依赖的法律结论。
