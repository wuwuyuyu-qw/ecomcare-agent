"""Provision synthetic local demo accounts, generating passwords once.

Run from python/: uv run python scripts/init_demo_auth.py
This script writes only the separate identity SQLite file, never business data.
"""
import argparse
import secrets
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.auth import create_demo_account


def main() -> None:
    parser = argparse.ArgumentParser(description="创建本地合成演示账号（密码随机生成）")
    parser.add_argument("--db-path", default="data/demo_auth.sqlite", help="与 AUTH_DB_PATH 相同，默认使用项目独立 SQLite")
    parser.add_argument("--username", help="只创建指定账号；省略则创建三个演示账号")
    parser.add_argument("--role", choices=("customer", "admin"), default="customer")
    args = parser.parse_args()
    accounts = [(args.username, args.role)] if args.username else [
        ("customer_demo", "customer"), ("customer_other", "customer"), ("operator_demo", "admin")]
    for username, role in accounts:
        password = secrets.token_urlsafe(18)
        try:
            create_demo_account(username, password, role, path=args.db_path)
        except sqlite3.IntegrityError:
            print(f"{username}: 已存在，保留原密码与会话")
            continue
        print(f"账号: {username} | 角色: {role} | 密码: {password}")
    print("密码只在本次输出；请保存在自己的密码管理器中，不要提交到仓库。")


if __name__ == "__main__":
    main()
