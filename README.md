# Deep Research Agent

一个可观测的多 Agent 深度研究系统（由戴总研发）。系统会把研究问题拆分为 DAG 子任务，并行执行检索、阅读和分析，最终合成带来源引用的研究报告。内置 Dashboard 可以查看任务数据流、Agent 循环、工具调用、上下文压缩及各子任务的上下游数据。

## 功能概览

- DAG 规划与多 Agent 并行研究
- Web、网页、ArXiv、文件、计算器等研究工具
- 会话记忆与跨会话知识库
- Red/Blue 对抗评审和报告修复
- 引用编号清洗、来源追踪与规则指标评测
- 完整事件流记录和可视化 Dashboard

## 环境要求

- Python 3.10 或更高版本
- Node.js 22.13 或更高版本（仅 Dashboard 需要）
- 至少一个兼容 OpenAI API 的模型后端
- Tavily 或秘塔搜索 Key（联网检索需要）

## 安装

建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

安装 Dashboard 依赖：

```powershell
cd dashboard
npm install
cd ..
```

## 配置

在项目根目录创建 `.env`。该文件已被 Git 忽略，不要提交 API Key。

```dotenv
# 默认研究模型
DEEPSEEK_API_KEY=your_deepseek_key

# config/default.yaml 当前还为 judge/compressor 指定了 mimo
MIMO_API_KEY=your_mimo_key
MIMO_BASE_URL=your_mimo_compatible_endpoint
MIMO_MODEL=your_mimo_model

# 搜索服务二选一，也可以同时配置
TAVILY_API_KEY=your_tavily_key
METASO_KEY=your_metaso_key
```

模型、并发、工具预算、记忆和对抗参数位于 `config/default.yaml`。模型环境变量遵循 `{BACKEND}_API_KEY`、`{BACKEND}_BASE_URL`、`{BACKEND}_MODEL` 的命名方式；需要更换后端时，同时修改 `model.backend` 和 `model.backend_mapping`。

## 运行研究

直接提交一个问题：

```powershell
python run.py --query "2026 年具身智能的技术路线与产业化进展"
```

启用 Red/Blue 对抗优化：

```powershell
python run.py --query "你的研究问题" --adversarial
```

不提供 `--query` 时进入交互模式，可以新建研究或继续历史会话：

```powershell
python run.py
```

使用其他配置文件：

```powershell
python run.py --query "你的研究问题" --config config/default.yaml
```

报告写入 `outputs/`，每次运行的事件、过程日志和中间状态位于 `outputs/debug_runs/<run_id>/`。

## Dashboard

一条命令同时启动本地 API 和前端：

```powershell
.\start_dashboard.ps1
```

- Dashboard：<http://localhost:3000>
- 本地 API：<http://127.0.0.1:8765>

脚本要求项目中存在 `.venv`，并要求 Node.js 22.13+。按 `Ctrl+C` 可停止服务。

## 测试与评测

运行全部单元测试：

```powershell
python -m pytest -q
```

运行无需模型 API 的引用解析评测：

```powershell
python eval_citation_parser.py --output outputs
```

运行一个指定的 ResearchBench 题目：

```powershell
python eval_json_parser_ablation.py --bench-id tech_001 --config config/experiment.yaml --output outputs/evaluation
```

固定种子抽样或运行全部题目：

```powershell
python eval_json_parser_ablation.py --sample 5 --seed 20261009 --config config/experiment.yaml --output outputs/evaluation
python eval_json_parser_ablation.py --run-all --config config/experiment.yaml --output outputs/evaluation
```

将评测汇总 JSON 聚合为统计报告：

```powershell
python scripts/aggregate_experiment.py --summary outputs/eval_json_ablation_summary_<timestamp>.json
```

端到端评测会调用外部模型和搜索服务，耗时与费用随题目数、子任务数和对抗轮数增长。实验配置使用独立的 `tmp/experiment/*.db`，不会污染日常记忆库。

## 已有测评结果

最新纳入版本控制的结果见 [15 题端到端评测报告](docs/evaluation/2026-10-09-end-to-end.md)。该次实验 15/15 完成，Red 对抗总体评分由 6.6387 提升至 7.0413；报告也记录了没有改善的规则指标和工具失败情况，便于客观判断系统边界。

## 项目结构

```text
src/deep_research/       核心编排、Agent、模型、工具、记忆和评测模块
config/                  日常与实验配置
dashboard/               可视化前端
scripts/                 评测聚合和维护脚本
tests/unit/              单元测试
docs/                    设计说明与版本化评测报告
outputs/                 本地报告和运行轨迹（默认不提交）
run.py                   研究入口
dashboard_server.py      Dashboard 本地 API
```

## 注意事项

- `.env`、`data/`、`tmp/` 和 `outputs/` 默认不会被 Git 跟踪。
- 研究报告可能包含模型生成错误；用于医疗、法律、金融等高风险场景前，应人工核验原始来源。
- 批量评测前建议先运行单题，确认模型、搜索服务和超时配置可用。
