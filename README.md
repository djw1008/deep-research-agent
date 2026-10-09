# Deep Research Agent

一个可观测的多 Agent 深度研究系统(由戴总研发)。系统会把问题拆成 DAG 子任务，并行执行检索与分析，合成研究报告；Dashboard 可查看数据流、Agent 循环、工具调用、上下文压缩和每个子任务的上下游数据。

## 本地运行

```bash
pip install -r requirements.txt
python run.py --query "你的研究问题"
```

启动可视化 Dashboard：

```powershell
.\start_dashboard.ps1
```

- Dashboard：`http://localhost:3000`
- 本地 API：`http://127.0.0.1:8765`

API Key 等敏感配置放在 `.env` 中；模型、并发、压缩和记忆配置位于 `config/default.yaml`。

## 测试

```bash
python -m pytest tests/unit -q
```
