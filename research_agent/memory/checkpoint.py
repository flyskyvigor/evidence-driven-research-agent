"""LangGraph SQLite Checkpointer 工厂。

SQLite 适合本项目的单机 Linux 演示和小规模部署。多进程生产环境应替换为
PostgresSaver。数据库文件必须被视为可信状态，不应接受不可信来源覆盖。
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from research_agent.config import get_checkpoint_db_path


def create_sqlite_checkpointer(path: str | Path | None = None) -> SqliteSaver:
    target = Path(path) if path else get_checkpoint_db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    connection = sqlite3.connect(str(target), check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver

