# 本地 Agent WebUI

在现有的 `test` Conda 环境启动，无需安装额外的 Web 框架：

```powershell
cd C:\PythonProject\my-agent
conda activate test
python webui.py
```

浏览器打开 http://127.0.0.1:7860 。终端按 Ctrl+C 停止服务。
也可以不激活环境，直接使用 Conda：

```powershell
conda run --no-capture-output -n test python webui.py
```

端口被占用时运行 `python webui.py --port 7861`。
`--session-db` 可指定会话 ID 数据库，默认复用命令行的 `.runtime/conversations.db`。

界面支持创建和切换记忆会话、发送多行消息、复制回答，以及展开或隐藏每轮的思考过程。
每个会话右侧的垃圾桶按钮可删除对话。确认后，后台复用 `MemoryTool._forget_with_session`
清空该 session 的所有已启用记忆（包括正文、向量和语义图来源），成功后删除会话 ID
及当前页面中的聊天内容；其他 session 的记忆保留。删除失败时显示错误并保留会话 ID
供重试，已经删除的记忆不会回滚。聊天运行期间禁止删除，记忆清理期间禁止新的聊天请求。
删除当前会话后切换至其余会话；删除最后一个会话后可以点击“新建会话”继续使用。
Enter 发送，Shift+Enter 换行；中文输入法选字时不会触发发送。
思考区逐步骤显示 agent 实际输出的 `Thought` 动作目的摘要、工具名称与参数、
`Observation` 工具返回；最终回答独立显示。当前接入按 ReAct 步骤更新，不是逐字输出，
也不读取模型服务没有提供的内部推理。工具结果过长时在面板内滚动查看。

停止按钮会在当前模型请求或工具调用结束后停止后续步骤，已经执行的操作不会撤销。
关闭页面也会请求停止；服务同时只执行一个请求，避免共用 agent 的会话绑定互相干扰。

聊天内容和步骤仅保存在当前页面内存，刷新或关闭后消失，不写入浏览器存储或聊天数据库。
切换会话可查看当前页面中该会话的消息；以前打开页面的聊天不会恢复。
模型上下文保持现有行为，只包含当前请求、本轮执行历史和工作记忆。
agent 按其现有记忆工具规则保存的记忆仍遵循原有逻辑。

后台复用 `my_main.build_agent` 和项目 `.env`；API key 只在服务端读取。
页面连接成功仅说明本地 Web 服务可用，模型和记忆模块会在首次发送时初始化。
模型服务或记忆组件配置有误时，错误会显示在该轮回答中。
服务仅监听 `127.0.0.1`，只允许页面资源和明确的 API 路由访问。

验证：

```powershell
$env:PYTHONPATH = "$PWD;$PWD\memory"
conda run --no-capture-output -n test python -m unittest discover -s tests -v
```
