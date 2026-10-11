# 会话 ID 与记忆检索

每个 session 只持久化 `session_id`，不保存聊天消息、首条问题摘要、时间或消息数量。
选择旧 session 只切换记忆所属 ID，不会恢复或向模型发送以前的问答。
同一 session 的下一轮也不会自动带上上一轮聊天内容。

在用户指定的 Conda 环境运行：

```powershell
conda run --no-capture-output -n test python my_main.py
```

启动时列出已有 ID；输入 ID 选择它，直接回车或输入 `new` 新建 ID。

| 命令 | 作用 |
| --- | --- |
| `/sessions` | 列出 session_id，`*` 标记当前会话 |
| `/use <session_id>` | 切换当前记忆会话 ID |
| `/new` | 创建并选择新的 session_id |
| `/help` | 显示命令 |
| `/exit` 或 `exit` | 退出 |

启动参数：

```powershell
conda run --no-capture-output -n test python my_main.py --list-sessions
conda run --no-capture-output -n test python my_main.py --session session_实际ID
conda run --no-capture-output -n test python my_main.py --new
```

ID 默认保存到 `.runtime/conversations.db`；`--session-db` 或环境变量
`AGENT_SESSION_DATABASE_PATH` 可指定其他数据库。会话注册表只有 `session_id` 一列。
旧格式数据库会在启动时迁移：保留所有 ID，删除已保存的聊天正文及摘要。
已有 working、episodic、semantic 记忆不因迁移被删除。

记忆工具的 add 自动把当前 ID 写入 `metadata["session_id"]`；search 自动限定当前 ID。
working、episodic、semantic 都先限制所属会话，再应用结果数量限制。
semantic 从 Neo4j 原始 metadata 筛选候选 ID，再限制图及向量两路召回，兼容旧记录。
没有 session_id 或属于其他 session 的记忆，不会出现在当前会话的工具搜索结果里。

需要此前信息时，模型应调用记忆工具 search。没有保存到记忆库的信息无法从历史问答恢复。
ReAct 本轮的 Action/Observation 只在本轮执行期间临时传给模型，结束后不保留。
`agent.get_history()` 始终返回空列表；旧的 `add_message`、`record_turn`、`clear_history`
仅作为兼容入口，不保存任何内容。

测试：

```powershell
conda run --no-capture-output -n test python -m unittest discover -s tests -v
```
