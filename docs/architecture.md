# 实际执行链与源码定位

根README的Mermaid省略投诉/话术和实验分支，以便快速阅读。以下入口对应实际代码，不把固定Workflow描述为全自主规划。

| 链路 | 源码入口 | 关键职责 |
| --- | --- | --- |
| 请求与身份 | [agent.py](../app/api/agent.py) `run_agent`、[auth.py](../app/core/auth.py) | Principal来自服务端，检查会话归属 |
| 轮次生命周期 | [runtime.py](../app/graph/runtime.py) `_bounded_run/_prepare_scope` | 单进程会话互斥、deadline、checkpoint、取消收尾 |
| 图组装 | [build.py](../app/graph/build.py) `_builder` | task_entry → resolve_reference → classify_intent → 分支 |
| 物流固定流程 | [logistics.py](../app/graph/logistics.py) `extract_logistics/investigate_logistics` | 诉求提取、核验订单、查询物流、程序规则与模板答复 |
| ReAct | [nodes.py](../app/graph/nodes.py) `main_agent/agent_tools` | 模型提出调用，结果回流ToolMessage，再生成或停止 |
| TaskContext | [task_context.py](../app/graph/task_context.py) `begin_turn/current_facts/guard_tool_call/record_tool_result` | claims与核验事实分离；用户/会话/任务/订单scope；默认TTL900秒；换单及失败撤销 |
| 工具注册 | [registry.py](../app/tools/registry.py) `get_all_specs/permission_for` | MCP发现后的本地允许表与结果契约要求；不信远端自报权限 |
| 执行与回流 | [engine.py](../app/tools/engine.py) `execute_tool_call` | 参数校验、可信身份注入、权限、deadline、只读有限重试、结果契约、ToolRun与审计 |
| MCP | [mcp_client.py](../app/tools/mcp_client.py)、[物流服务](../mcp_servers/logistics_server.py) | 本地Streamable HTTP服务，返回Mock物流；不是独立完整的身份边界 |
| RAG | [retrieval.py](../app/core/retrieval.py) `search_knowledge`、[milvus_client.py](../app/kb/milvus_client.py) | Dense与中文BM25召回 → SDK RRFRanker → 外部Rerank；query_faq工具也调用该链 |
| 政策核验 | [policy_evidence.py](../app/core/policy_evidence.py) `evaluate_citations` | 来源审批、版本、正文哈希、时间/地域/履约及有限条款条件；不足则UNKNOWN |
| Memory | [memory.py](../app/core/memory.py)、[summarizer.py](../app/core/summarizer.py) | 历史、滑窗、分层压缩和摘要；摘要不是订单授权或当前事实 |
| 输出证据 | [answer_guard.py](../app/graph/answer_guard.py) `guard_answer` | 当前scope、必要证据、明显未执行写入断言的有限规则检查 |
| 评测/Usage | [dev_batch.py](../app/evaluation/dev_batch.py)、[usage_meter.py](../app/evaluation/usage_meter.py)、[money_guard.py](../app/evaluation/money_guard.py) | 物理HTTP含重试、请求Usage、费用已知/未知与停止契约 |

## 必须解释的例外

- `nodes.fetch_order` 是可信直读路径，不通过统一工具引擎。
- 物流Workflow以自身授权、规则及模板完成输出，不经过通用answer_guard。
- TaskContext在Graph层约束业务依赖；工具引擎本身不是完整任务状态机。
- MySQL保存消息与审计，SQLite保存checkpoint和演示认证。它们不组成跨库事务；当前会话锁也不是分布式锁。
- 已存在的工单/退款相关模块不等于当前开放写入：API/Graph安全开关保持关闭，仅选择订单等受控恢复开放。
