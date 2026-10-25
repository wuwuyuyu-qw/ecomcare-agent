# 课程基础与二次开发贡献

项目来源与公开许可见 [COPYRIGHT](../COPYRIGHT.md)。课程比较基准是原私有仓库章节提交 `90e4b2465b45ee99ebc9792e588108a2f3646275`。公开仓库从新历史开始，不把课程章节提交移植为个人提交。

| 主题 | 课程已提供 | 本人参与的增量（Codex辅助） |
| --- | --- | --- |
| LangGraph | Workflow/ReAct、意图、指代、interrupt/resume | 固定物流调查、只读边界、任务依赖和受控恢复 |
| Memory/Context | 滑窗、摘要、消息/checkpoint | TaskContext事实来源、scope/TTL、换单及核验失败失效 |
| Tool/MCP | 注册、动态发现、统一执行与审计 | 远端本地允许表、身份注入、严格返回契约、deadline/取消等约束 |
| RAG | Dense/BM25/RRF/Rerank、Query改写与基础证据检查 | 政策审批/版本/正文绑定/适用条件核验；未证明正式业务资格判断 |
| 评测观测 | Langfuse、课程评测与成本账 | 冻结任务、Graph/Tool观察、请求级Usage、预算保护及失败审计 |
| 累计Usage | `_CumulativeUsageChatOpenAI`和累计差分 | 单次流隔离状态，修复跨调用复用少计；不是整类封装从零实现 |

## 三个面试主讲点

1. **事实有效性。** 用户声明不能授权查单。核验事实带来源、时间与scope，换单或复核失败撤销依赖，避免旧物流结论延续到新单。[TaskContext](../app/graph/task_context.py)。
2. **工具执行约束。** 不靠Prompt约定权限，使用可信身份、MCP允许表、输入输出契约和deadline。输出Guard是有限规则，不能声称通用防注入。[Engine](../app/tools/engine.py)、[Registry](../app/tools/registry.py)。
3. **可信观测与失败保留。** 流式累计差分原属于共享模型实例；按单次流隔离。免费重排价格为0也不能把未知Token填0，缺测后停止后续请求。历史业务PASS与保护FAIL分别保留。[LLM](../app/core/llm.py)、[Money Guard](../app/evaluation/money_guard.py)。

这些改动使用Codex辅助实现、审查、测试与文档整理。原私有工作区没有后续个人Git提交可独立证明逐行手写归属，不能把代码存在等同于完全自主实现。本人需能够解释所述工作、取舍和验证边界。

## 本次公开整理的额外改动

- 只在独立目录制作全新Git历史，保留原作者署名。
- 数据库连接改为显式配置，移除共享默认口令；独立Compose与随机本地凭据生成，避免冲突已有容器。
- 原shell入口改为等待Compose健康，移除固定9091探测，匹配公开目录的独立端口配置。
- 课程启发参考评测原从旧Git读取Prompt。公开目录不携带旧Git对象，故增加 [来源快照](../provenance/course-agent-system.json) 与严格校验读取；Prompt原文与只读后缀不变，评分器不变。
- 课程知识正文、学习文档及原始报告不分发，换成自写合成知识示例；新语料不继承历史运行成绩。

当前没有重构生产Graph、Guard、模型配置语义、业务Fixture或评分答案来提高分数。公开打包变更与历史修复通过来源清单分开记录。
