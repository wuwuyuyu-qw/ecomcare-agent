#!/usr/bin/env bash
# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
# 单入口拉起 FastAPI 应用(:8000);三组上游各自直连,没有网关进程
set -euo pipefail
cd "$(dirname "$0")/.."

# Python 强制 UTF-8。中文 Windows 默认编码是 GBK,读本仓库里带中文的 markdown / jsonl
# (知识库、评估集)会抛 UnicodeDecodeError。这个变量必须在 Python 启动前设好,写进 .env 太晚。
export PYTHONUTF8=1

if [ ! -f .env ]; then
  echo "缺少 .env,请 cp .env.example .env 并填入 token" >&2
  exit 1
fi

# Public packaging uses an isolated Compose project and configurable host ports.
# Compose reads .env as data; never source a credentials file as shell code.
echo "等待本项目依赖容器健康..."
docker compose up -d --wait mysql etcd minio milvus-standalone
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --locked python scripts/run_local_agent.py --env-file "$PWD/.env"
