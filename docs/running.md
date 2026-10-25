# Windows / WSL 运行与测试

## 环境

Python 3.12、uv、Docker Desktop（WSL集成）。推荐在VSCode的WSL终端运行；仓库可放在任意目录，不依赖作者机器路径。`pyproject.toml` / `uv.lock` 保留原运行时依赖，没有升级依赖。

Docker Compose 使用独立项目 `ecomcare-local` 与新数据卷，默认 MySQL 13307、Milvus 29530、健康端口 19091，均只映射到本机。API 8000，MCP 8101/8102。端口占用时检查实际进程，不停止或清理无关容器。

## WSL（推荐）

按根README逐条执行。`init_local_env.py` 使用独占创建，不覆盖已有 `.env`。填自己的服务配置后，`docker compose up -d` 首次空卷依文件名执行 `sql/`；不要对已有库重复执行全部建表DDL。

`scripts/build_kb.py` 读取 `data/kb` 的6份自写合成资料，写入MySQL pending块；**`scripts/vectorize_kb.py` 会调用嵌入服务**。没有付费授权时停在此步骤，不把“建了SQL记录”当作已建好向量索引。新知识示例不包含真实商家政策，不能继承历史评测结果。

`init_demo_auth.py` 在独立SQLite创建 `customer_demo` / `customer_other` / `operator_demo`，随机密码只输出一次。不要投屏初始化过程、日志Headers或 `.env`。

`MEWHELP_FIXTURE_PROFILE` 必须是进程环境变量，启动器把它传给API与MCP。只把它写进 `.env` 不足以保证两个进程使用同一世界。

## PowerShell 原生环境的等价命令

以下要求本机已安装兼容的Python3.12、uv及Docker；**本轮没有将Windows原生服务作为验收平台**。

```powershell
uv sync --locked --group dev --no-group ml
uv run --locked python scripts/init_local_env.py
$env:PYTHONPATH = (Get-Location).Path
# 私下编辑.env填入自己的供应商配置，不在终端输出内容。
docker compose up -d
docker compose ps
uv run --locked python scripts/build_kb.py
# 下一条会调用嵌入供应商，操作者确认费用后再手动执行。
uv run --locked python scripts/vectorize_kb.py
uv run --locked python scripts/init_demo_auth.py
$env:MEWHELP_FIXTURE_PROFILE = 'task-v1'
uv run --locked python scripts/run_local_agent.py --env-file "$PWD/.env"
```

浏览器进入 <http://127.0.0.1:8000>。结束按Ctrl+C；启动器只关闭自己创建的API/MCP，不删除容器和卷。主进程保持单worker，当前不支持靠增加worker保证跨实例会话互斥。

## 零供应商请求验证

```powershell
uv run --locked --group dev --no-group ml python -B scripts/check_agent_core.py --tests-only --compatible-course
```

离线入口清除子进程供应商配置并阻止网络、业务数据库、`.env`和旧留出标签。它只执行指定允许表，不是全项目覆盖率。依赖下载不等于模型推理，本轮复用已有环境；`uv lock --check --offline` 已验证锁文件可接受。

完整课程pytest包含测试库重建和旧写操作断言，不能直接对日常库执行。部分课程专用评测素材未公开，相关历史脚本不能当作当前可复现成绩。

## 最短排障顺序

1. `docker compose ps`：等待MySQL与Milvus健康。MySQL不可达先核对端口和 `.env`，不直接重建卷。
2. `http://127.0.0.1:19091/healthz`：检查新Compose的Milvus；本机已有实例端口可能不同。
3. `http://127.0.0.1:8000/readyz`：200仅表示图、数据库读取和启动知识快照就绪，不代表供应商持续可用。
4. MCP握手失败：检查8101/8102和profile；不要把业务MCP直接暴露到公网。
5. 401重新私下登录；409等待同会话上一轮完成；502/504停止发问并检查失败层，不连续重试直至偶然成功。

模型请求是手动触发；手动聊天入口没有自动金额上限。评测预算保护仅在明确安装闸门的受控评测进程生效。
